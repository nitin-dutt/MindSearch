"""Authentication (alignment_plan SEC-01): signed tokens and salted password hashes.

Standard library + `cryptography` only, so the existing images need no new dependency.
  * Tokens are HS256 JWTs (RFC 7519). Verification pins the algorithm (so the classic `alg: none` and
    algorithm-confusion attacks fail), compares signatures in constant time, and enforces expiry.
  * Passwords are stored as scrypt hashes with a per-user random salt, never in plaintext.
  * The signing secret must come from the environment; there is no default.

If PyJWT/passlib are added to the image later, swapping them in only touches this file.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional

import yaml
from cryptography.exceptions import InvalidKey
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

ALGORITHM = "HS256"
MIN_SECRET_CHARS = 32
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1


class AuthError(Exception):
    """Any authentication failure. The message is safe to log, not to return to a client."""


# ---------------------------------------------------------------- secrets

def load_secret(env_var="MINDSEARCH_JWT_SECRET") -> str:
    secret = os.getenv(env_var, "")
    if len(secret) < MIN_SECRET_CHARS:
        raise AuthError(
            f"{env_var} must be set to a random string of at least {MIN_SECRET_CHARS} characters "
            f"(generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\")")
    return secret


# ------------------------------------------------------------------ tokens

def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(signing_input: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()


def create_token(claims: dict, secret: str, ttl_seconds: int = 3600, now: Optional[float] = None) -> str:
    now = int(now if now is not None else time.time())
    payload = {**claims, "iat": now, "exp": now + int(ttl_seconds)}
    header = {"alg": ALGORITHM, "typ": "JWT"}
    signing_input = f"{_b64e(json.dumps(header, separators=(',', ':')).encode())}." \
                    f"{_b64e(json.dumps(payload, separators=(',', ':')).encode())}"
    return f"{signing_input}.{_b64e(_sign(signing_input.encode(), secret))}"


def verify_token(token: str, secret: str, now: Optional[float] = None) -> dict:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64d(header_b64))
        payload = json.loads(_b64d(payload_b64))
        signature = _b64d(sig_b64)
    except Exception as e:  # noqa: BLE001 - any malformed token is simply invalid
        raise AuthError("malformed token") from e
    if header.get("alg") != ALGORITHM:           # rejects alg=none and algorithm confusion
        raise AuthError("unexpected token algorithm")
    expected = _sign(f"{header_b64}.{payload_b64}".encode(), secret)
    if not hmac.compare_digest(signature, expected):
        raise AuthError("bad signature")
    current = now if now is not None else time.time()
    if "exp" not in payload or current >= payload["exp"]:
        raise AuthError("token expired")
    return payload


# --------------------------------------------------------------- passwords

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P).derive(password.encode())
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64e(salt)}${_b64e(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        Scrypt(salt=_b64d(salt_b64), length=32, n=int(n), r=int(r), p=int(p)).verify(
            password.encode(), _b64d(digest_b64))
        return True
    except (InvalidKey, ValueError):
        return False


# -------------------------------------------------------------- user store

@dataclass(frozen=True)
class UserRecord:
    username: str
    password_hash: str
    level: str
    groups: tuple = ()
    roles: tuple = ()


class UserStore:
    """Users from a YAML file (users.yaml, git-ignored; create entries with manage_users.py)."""

    def __init__(self, records):
        self._by_name = {r.username: r for r in records}
        # used to spend the same time on unknown usernames, so timing does not reveal which exist
        self._dummy_hash = hash_password(secrets.token_urlsafe(16))

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls([UserRecord(u["username"], u["password_hash"], u["level"], tuple(u.get("groups", ())),
                               tuple(u.get("roles", ()))) for u in data.get("users", [])])

    def records(self):
        return list(self._by_name.values())

    def get(self, username) -> Optional[UserRecord]:
        return self._by_name.get(username)

    def authenticate(self, username, password) -> Optional[UserRecord]:
        record = self._by_name.get(username)
        ok = verify_password(password, record.password_hash if record else self._dummy_hash)
        return record if (record and ok) else None
