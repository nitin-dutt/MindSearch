import base64
import json

import pytest

import auth
from auth import AuthError, UserRecord, UserStore, create_token, hash_password, verify_password, verify_token

SECRET = "x" * 40


def forge(header, payload, secret=SECRET, sign=True):
    """Build a token by hand so attacks can be tested."""
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    signing_input = f"{b(header)}.{b(payload)}"
    sig = auth._b64e(auth._sign(signing_input.encode(), secret)) if sign else ""
    return f"{signing_input}.{sig}"


def test_valid_token_round_trips_claims():
    token = create_token({"sub": "alice", "level": "Manager"}, SECRET, ttl_seconds=60, now=1000)
    claims = verify_token(token, SECRET, now=1010)
    assert claims["sub"] == "alice" and claims["level"] == "Manager"


def test_expired_token_is_rejected():
    token = create_token({"sub": "a"}, SECRET, ttl_seconds=60, now=1000)
    with pytest.raises(AuthError, match="expired"):
        verify_token(token, SECRET, now=1060)


def test_tampered_payload_is_rejected():
    """The attack the old design allowed: change your own role. Signature must not match."""
    token = create_token({"sub": "emp", "level": "Employee"}, SECRET, now=1000)
    header, payload, sig = token.split(".")
    evil = auth._b64e(json.dumps({"sub": "emp", "level": "Admin", "exp": 9999999999}).encode())
    with pytest.raises(AuthError, match="signature"):
        verify_token(f"{header}.{evil}.{sig}", SECRET, now=1001)


def test_token_signed_with_another_secret_is_rejected():
    token = create_token({"sub": "a"}, "y" * 40, now=1000)
    with pytest.raises(AuthError, match="signature"):
        verify_token(token, SECRET, now=1001)


def test_alg_none_attack_is_rejected():
    token = forge({"alg": "none", "typ": "JWT"}, {"sub": "admin", "exp": 9999999999}, sign=False)
    with pytest.raises(AuthError, match="algorithm"):
        verify_token(token, SECRET, now=1)


def test_other_algorithms_are_rejected_even_with_a_valid_hmac():
    token = forge({"alg": "HS512", "typ": "JWT"}, {"sub": "a", "exp": 9999999999})
    with pytest.raises(AuthError, match="algorithm"):
        verify_token(token, SECRET, now=1)


def test_token_without_expiry_is_rejected():
    token = forge({"alg": "HS256", "typ": "JWT"}, {"sub": "a"})
    with pytest.raises(AuthError, match="expired"):
        verify_token(token, SECRET, now=1)


@pytest.mark.parametrize("junk", ["", "abc", "a.b", "a.b.c.d", "....", "not a token at all"])
def test_malformed_tokens_are_rejected(junk):
    with pytest.raises(AuthError):
        verify_token(junk, SECRET)


def test_secret_must_be_long_and_present(monkeypatch):
    monkeypatch.delenv("MINDSEARCH_JWT_SECRET", raising=False)
    with pytest.raises(AuthError):
        auth.load_secret()
    monkeypatch.setenv("MINDSEARCH_JWT_SECRET", "short")
    with pytest.raises(AuthError):
        auth.load_secret()
    monkeypatch.setenv("MINDSEARCH_JWT_SECRET", "z" * 40)
    assert auth.load_secret() == "z" * 40


def test_password_hash_is_salted_and_verifies():
    h1, h2 = hash_password("correct horse"), hash_password("correct horse")
    assert h1 != h2                              # different salt each time
    assert "correct horse" not in h1
    assert verify_password("correct horse", h1) and verify_password("correct horse", h2)
    assert not verify_password("wrong", h1)
    assert not verify_password("correct horse", "plaintext-not-a-hash")


def test_authenticate_accepts_only_the_right_user_and_password():
    store = UserStore([UserRecord("alice", hash_password("pw-alice"), "Manager", ("HR",), ("Admin",))])
    assert store.authenticate("alice", "pw-alice").level == "Manager"
    assert store.authenticate("alice", "nope") is None
    assert store.authenticate("mallory", "pw-alice") is None      # unknown user
