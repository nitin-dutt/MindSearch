"""Create, list and remove MindSearch users (alignment_plan SEC-01).

    python manage_users.py add alice --level Manager --groups HR,Legal            # prompts for the password
    python manage_users.py add root  --level Employee --roles Admin
    python manage_users.py list
    python manage_users.py remove alice

In Docker:  docker compose exec -it celery_worker python manage_users.py add alice --level Manager
(-it is needed so the password prompt works.) Levels, groups and super roles are validated against the
active org policy (org_policy.yaml). Users live in users.yaml (git-ignored); passwords are stored only
as salted scrypt hashes. The API reads the file at startup, so restart it after changes.
"""
import argparse
import getpass
import os
import sys

import yaml

from access_policy import AccessPolicy, PolicyError
from auth import hash_password

DEFAULT_FILE = os.getenv("USERS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.yaml"))
MIN_PASSWORD_CHARS = 10


def load(path):
    if not os.path.exists(path):
        return {"users": []}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"users": []}


def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    try:
        os.chmod(path, 0o600)       # no-op on Windows
    except OSError:
        pass


def add_user(data, policy, username, password, level, groups=(), roles=()):
    """Validate against the policy and add or replace the user. Returns the updated data."""
    if len(password) < MIN_PASSWORD_CHARS:
        raise PolicyError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
    policy.make_user(username, level, groups, roles)          # raises PolicyError on unknown level/group
    entry = {"username": username, "password_hash": hash_password(password), "level": level,
             "groups": list(groups), "roles": list(roles)}
    data["users"] = [u for u in data.get("users", []) if u["username"] != username] + [entry]
    return data


def remove_user(data, username):
    before = len(data.get("users", []))
    data["users"] = [u for u in data.get("users", []) if u["username"] != username]
    return len(data["users"]) < before


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_FILE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--level", required=True)
    a.add_argument("--groups", default="", help="comma separated")
    a.add_argument("--roles", default="", help="comma separated, e.g. Admin (super roles come from the policy)")
    sub.add_parser("list")
    r = sub.add_parser("remove")
    r.add_argument("username")
    args = ap.parse_args(argv)

    data = load(args.file)
    if args.cmd == "list":
        for u in data["users"]:
            print(f"{u['username']:<20} level={u['level']:<14} groups={','.join(u.get('groups', [])) or '-':<24} roles={','.join(u.get('roles', [])) or '-'}")
        return 0
    if args.cmd == "remove":
        if not remove_user(data, args.username):
            sys.exit(f"No such user: {args.username}")
        save(args.file, data)
        print(f"Removed {args.username}")
        return 0

    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match")
    try:
        add_user(data, AccessPolicy.from_yaml(), args.username, password, args.level,
                 [g for g in args.groups.split(",") if g], [r for r in args.roles.split(",") if r])
    except PolicyError as e:
        sys.exit(f"Error: {e}")
    save(args.file, data)
    print(f"Saved {args.username} to {args.file}. Restart the API to load it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
