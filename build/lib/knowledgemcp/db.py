"""Two ways to run the same SQL.

- DataApiExecutor: Aurora via the RDS Data API (HTTPS). Lambda needs no VPC and no NAT gateway.
- PsycopgExecutor: a normal Postgres connection, for local development and integration tests.

SQL uses ':name' parameters (Data API style). PsycopgExecutor converts them to '%(name)s'.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

_PARAM = re.compile(r"(?<!:):([a-zA-Z_][a-zA-Z0-9_]*)")


class Executor(Protocol):
    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def execute_many(self, sql: str, param_sets: list[dict[str, Any]]) -> None: ...


def to_pyformat(sql: str) -> str:
    sql = sql.replace("%", "%%")
    return _PARAM.sub(lambda m: f"%({m.group(1)})s", sql)


def data_api_param(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        field = {"isNull": True}
    elif isinstance(value, bool):
        field = {"booleanValue": value}
    elif isinstance(value, int):
        field = {"longValue": value}
    elif isinstance(value, float):
        field = {"doubleValue": value}
    else:
        field = {"stringValue": str(value)}
    return {"name": name, "value": field}


class DataApiExecutor:
    # Aurora Serverless v2 with min 0 ACU pauses when idle. The first call after a pause fails with
    # DatabaseResumingException while it wakes up (~15 s), so we retry with backoff.
    RETRYABLE = ("DatabaseResumingException", "StatementTimeoutException", "ServiceUnavailableError")

    def __init__(self, client, cluster_arn: str, secret_arn: str, database: str,
                 max_wait_s: float = 60.0, sleep=time.sleep):
        self.client = client
        self.cluster_arn = cluster_arn
        self.secret_arn = secret_arn
        self.database = database
        self.max_wait_s = max_wait_s
        self.sleep = sleep

    def _call(self, fn, **kwargs):
        waited, delay = 0.0, 1.0
        while True:
            try:
                return fn(resourceArn=self.cluster_arn, secretArn=self.secret_arn,
                          database=self.database, **kwargs)
            except Exception as exc:  # botocore ClientError
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
                if code not in self.RETRYABLE or waited >= self.max_wait_s:
                    raise
                self.sleep(delay)
                waited += delay
                delay = min(delay * 2, 8.0)

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        resp = self._call(
            self.client.execute_statement,
            sql=sql,
            parameters=[data_api_param(k, v) for k, v in (params or {}).items()],
            formatRecordsAs="JSON",
        )
        return json.loads(resp.get("formattedRecords") or "[]")

    def execute_many(self, sql: str, param_sets: list[dict[str, Any]]) -> None:
        for start in range(0, len(param_sets), 100):  # keep each request small
            batch = param_sets[start : start + 100]
            self._call(
                self.client.batch_execute_statement,
                sql=sql,
                parameterSets=[[data_api_param(k, v) for k, v in p.items()] for p in batch],
            )


class PsycopgExecutor:
    def __init__(self, dsn: str):
        import psycopg
        from psycopg.rows import dict_row

        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(to_pyformat(sql), params or {})
            return list(cur.fetchall()) if cur.description else []

    def execute_many(self, sql: str, param_sets: list[dict[str, Any]]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(to_pyformat(sql), param_sets)
