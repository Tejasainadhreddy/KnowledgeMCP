"""Create an API key for a user and the teams they may read.

Local (prints a JSON entry to put in API_KEYS_JSON):
    python scripts/create_api_key.py --user asha --teams hr,finance
On AWS (adds the hashed key to the Secrets Manager secret created by CDK):
    python scripts/create_api_key.py --user asha --teams hr,finance --secret-id <ApiKeysSecretArn output>

The plain key is printed once. Only its SHA-256 hash is stored.
"""

import argparse
import json

from knowledgemcp.auth import hash_key, new_key


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--teams", default="", help="comma-separated, e.g. hr,finance (public is always included)")
    p.add_argument("--secret-id", help="Secrets Manager secret ARN/name to update")
    args = p.parse_args()

    key = new_key()
    entry = {"user_id": args.user, "teams": [t.strip().lower() for t in args.teams.split(",") if t.strip()]}
    if args.secret_id:
        import boto3

        sm = boto3.client("secretsmanager")
        current = json.loads(sm.get_secret_value(SecretId=args.secret_id)["SecretString"] or "{}")
        current[hash_key(key)] = entry
        sm.put_secret_value(SecretId=args.secret_id, SecretString=json.dumps(current))
        print(f"Added key for {args.user} to {args.secret_id}")
    else:
        print("Add this to API_KEYS_JSON:")
        print(json.dumps({hash_key(key): entry}))
    print(f"\nAPI key (shown once): {key}")


if __name__ == "__main__":
    main()
