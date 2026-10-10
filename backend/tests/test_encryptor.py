import base64

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import encryptor as E
from encryptor import Encryptor, EncryptionKeyError, record_aad

KEY = AESGCM.generate_key(bit_length=256)


def test_round_trip_and_fresh_nonce_each_time():
    enc = Encryptor(KEY)
    a, b = enc.encrypt("secret chunk"), enc.encrypt("secret chunk")
    assert a != b and a[:12] != b[:12]                    # random nonce: same text never encrypts the same
    assert enc.decrypt(a) == enc.decrypt(b) == "secret chunk"
    assert b"secret" not in a


def test_wrong_key_cannot_decrypt():
    blob = Encryptor(KEY).encrypt("x")
    with pytest.raises(InvalidTag):
        Encryptor(AESGCM.generate_key(bit_length=256)).decrypt(blob)


def test_flipped_bit_is_detected():
    enc = Encryptor(KEY)
    blob = bytearray(enc.encrypt("hello"))
    blob[-1] ^= 1
    with pytest.raises(InvalidTag):
        enc.decrypt(bytes(blob))


def test_associated_data_must_match_to_decrypt():
    enc = Encryptor(KEY)
    aad = record_aad("point-1", {"min_rank": 2, "groups": ["HR"], "users": []})
    blob = enc.encrypt("restricted", aad=aad)
    assert enc.decrypt(blob, aad=aad) == "restricted"
    for other in (record_aad("point-2", {"min_rank": 2, "groups": ["HR"], "users": []}),   # moved to another record
                  record_aad("point-1", {"min_rank": 0, "groups": ["HR"], "users": []}),   # access rule lowered
                  record_aad("point-1", {"min_rank": 2, "groups": [], "users": []}),       # department removed
                  record_aad("point-1", {"min_rank": 2, "groups": ["HR"], "users": ["eve"]}),  # extra reader added
                  None):
        with pytest.raises(InvalidTag):
            enc.decrypt(blob, aad=other)


def test_aad_ignores_list_order_but_not_content():
    a = record_aad("p", {"min_rank": 1, "groups": ["B", "A"], "users": []})
    b = record_aad("p", {"min_rank": 1, "groups": ["A", "B"], "users": []})
    assert a == b


def test_there_is_no_automatic_key_creation(monkeypatch, tmp_path):
    monkeypatch.delenv(E.KEY_ENV, raising=False)
    monkeypatch.delenv(E.KEY_FILE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(EncryptionKeyError, match="new-key"):
        Encryptor()
    assert list(tmp_path.iterdir()) == []                  # nothing was written to disk


def test_key_loaded_from_environment_or_secret_file(monkeypatch, tmp_path):
    key_b64 = base64.urlsafe_b64encode(KEY).decode()
    monkeypatch.setenv(E.KEY_ENV, key_b64)
    assert Encryptor().decrypt(Encryptor(KEY).encrypt("hi")) == "hi"
    monkeypatch.delenv(E.KEY_ENV)
    secret = tmp_path / "data_key"
    secret.write_text(key_b64 + "\n")
    monkeypatch.setenv(E.KEY_FILE_ENV, str(secret))
    assert Encryptor().decrypt(Encryptor(KEY).encrypt("hi")) == "hi"


@pytest.mark.parametrize("bad", ["not-base64!!", base64.urlsafe_b64encode(b"short").decode()])
def test_malformed_keys_are_rejected(monkeypatch, bad):
    monkeypatch.setenv(E.KEY_ENV, bad)
    with pytest.raises(EncryptionKeyError):
        Encryptor()


def test_generated_keys_are_valid_and_unique():
    k1, k2 = E.generate_key_b64(), E.generate_key_b64()
    assert k1 != k2 and len(E._decode_key(k1)) == 32
