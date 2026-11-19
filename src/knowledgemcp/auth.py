"""API-key authentication. Each key maps to a user and the teams they may read.

Keys are never stored in plain text: we keep SHA-256 hashes, so leaking the secret does not leak keys.
Format of the stored JSON:  {"<sha256 of key>": {"user_id": "asha", "teams": ["hr", "finance"]}}
"""

from __future__ import annotations

import hashlib
import json
import secrets

from .config import Settings
from .models import User

KEY_PREFIX = "kmcp_"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


class ApiKeyAuth:
    def __init__(self, entries: dict[str, dict]):
        self.entries = entries

    def authenticate(self, key: str | None) -> User | None:
        if not key or not key.startswith(KEY_PREFIX):
            return None
        entry = self.entries.get(hash_key(key))
        if not entry:
            return None
        return User(user_id=entry["user_id"], teams=[t.lower() for t in entry.get("teams", [])])


def load_auth(s: Settings, secrets_client=None) -> ApiKeyAuth:
    if s.api_keys_secret_arn:
        import boto3

        client = secrets_client or boto3.client("secretsmanager", region_name=s.region)
        raw = client.get_secret_value(SecretId=s.api_keys_secret_arn)["SecretString"]
    else:
        raw = s.api_keys_json or "{}"
    return ApiKeyAuth(json.loads(raw))
