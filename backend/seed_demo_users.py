"""Create demo users for LOCAL testing only.

    python backend/seed_demo_users.py

Creates demo_employee, demo_manager and demo_admin in users.yaml (git-ignored) with RANDOM passwords, and
writes the passwords to demo_credentials.txt (also git-ignored). Existing non-demo users are left alone.
Never use these accounts outside a local machine. For real accounts use manage_users.py.
"""
import os
import secrets
import sys

from access_policy import AccessPolicy
from manage_users import DEFAULT_FILE, add_user, load, save

HERE = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(HERE, "demo_credentials.txt")

# (username, level, roles); levels come from the active org policy (org_policy.yaml)
DEMO_USERS = [
    ("demo_employee", "Employee", []),
    ("demo_manager", "Manager", []),
    ("demo_admin", "Employee", ["Admin"]),
]


def main():
    policy = AccessPolicy.from_yaml()
    data = load(DEFAULT_FILE)
    lines = ["Demo credentials for local testing only (this file is git-ignored). Delete it when done.", ""]
    for username, level, roles in DEMO_USERS:
        password = secrets.token_urlsafe(12)
        add_user(data, policy, username, password, level, [], roles)
        lines.append(f"{username:<14} {password:<18} level={level} roles={','.join(roles) or '-'}")
    save(DEFAULT_FILE, data)
    with open(CREDENTIALS_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(DEMO_USERS)} demo users to {DEFAULT_FILE}; passwords are in {CREDENTIALS_FILE}")


if __name__ == "__main__":
    sys.exit(main())
