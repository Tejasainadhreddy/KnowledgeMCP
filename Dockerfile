# One image for all Lambda handlers (API, ingest, migrate); CDK picks the handler with `cmd`.
# Also runs locally with docker compose (uvicorn entrypoint).
FROM public.ecr.aws/lambda/python:3.12

COPY pyproject.toml README.md /tmp/build/
COPY src /tmp/build/src
RUN pip install --no-cache-dir "/tmp/build[local]" --target "${LAMBDA_TASK_ROOT}" \
    && rm -rf /tmp/build

CMD ["knowledgemcp.api.handler"]
