"""AES-GCM-256 encryption of stored chunks (alignment_plan SEC-05, SEC-07).

Key handling (SEC-05): the key is NEVER generated automatically and NEVER stored next to the data.
It is read from the environment (MINDSEARCH_DATA_KEY, base64) or from a mounted secret file
(MINDSEARCH_DATA_KEY_FILE, e.g. a Docker secret). Create one with:  python encryptor.py new-key

Integrity binding (SEC-07): every ciphertext is authenticated together with "associated data" naming its
record (Qdrant point id) and its access rule (min_rank, groups, users). A ciphertext copied onto another
record, or a record whose access rule was edited in the database, fails to decrypt instead of being served.
"""
import base64
import os
import sys

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_ENV = "MINDSEARCH_DATA_KEY"
KEY_FILE_ENV = "MINDSEARCH_DATA_KEY_FILE"


class EncryptionKeyError(Exception):
    pass


def generate_key_b64() -> str:
    return base64.urlsafe_b64encode(AESGCM.generate_key(bit_length=256)).decode()


def _decode_key(text: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode(text.strip() + "=" * (-len(text.strip()) % 4))
    except Exception as e:  # noqa: BLE001
        raise EncryptionKeyError("The data key is not valid base64") from e
    if len(key) != 32:
        raise EncryptionKeyError(f"The data key must decode to 32 bytes (AES-256), got {len(key)}")
    return key


def load_key() -> bytes:
    if os.getenv(KEY_ENV):
        return _decode_key(os.environ[KEY_ENV])
    path = os.getenv(KEY_FILE_ENV)
    if path:
        with open(path, encoding="utf-8") as f:
            return _decode_key(f.read())
    raise EncryptionKeyError(
        f"No data encryption key. Set {KEY_ENV} (or {KEY_FILE_ENV}). Generate one with: python encryptor.py new-key")


def record_aad(point_id, acl_payload: dict) -> bytes:
    """The associated data that binds a ciphertext to its Qdrant point and its access rule."""
    groups = ",".join(sorted(acl_payload.get("groups", [])))
    users = ",".join(sorted(acl_payload.get("users", [])))
    return f"{point_id}|{acl_payload.get('min_rank')}|{groups}|{users}".encode()


class Encryptor:
    def __init__(self, key: bytes = None):
        """AES-256-GCM. `key` must be 32 bytes; if omitted it is loaded from the environment / secret file."""
        if key is None:
            key = load_key()
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self.aesgcm = AESGCM(key)

    def encrypt(self, plaintext: str, aad: bytes = None) -> bytes:
        """Returns nonce (12 bytes) + ciphertext + tag. `aad` is authenticated but not stored."""
        nonce = os.urandom(12)  # NIST recommended nonce length
        return nonce + self.aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)

    def decrypt(self, data: bytes, aad: bytes = None) -> str:
        """Raises cryptography.exceptions.InvalidTag if the data, the key or the associated data differ."""
        return self.aesgcm.decrypt(data[:12], data[12:], aad).decode("utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["new-key"]:
        print(generate_key_b64())
    elif len(sys.argv) == 3 and sys.argv[1] == "from-file":
        # one-off migration: print a legacy raw 32-byte key file (encryption.key) in the new base64 form
        with open(sys.argv[2], "rb") as f:
            print(base64.urlsafe_b64encode(f.read()).decode())
    else:
        sys.exit("usage: python encryptor.py new-key | python encryptor.py from-file <legacy encryption.key>")
