import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Encryptor:
    def __init__(self, key=None):
        """
        Initialize AES-GCM (AES-256).
        Key must be 32 bytes (256 bits).
        If no key provided, generates a new one.
        """
        if key is None:
            self.key = AESGCM.generate_key(bit_length=256)
        else:
            if len(key) != 32:
                raise ValueError("Key must be 32 bytes for AES-256")
            self.key = key
            
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, plaintext: str) -> bytes:
        """
        Encrypts plaintext string to bytes.
        Returns: nonce + ciphertext + tag (handled by AESGCM)
        """
        nonce = os.urandom(12)  # NIST recommended nonce length
        data = plaintext.encode('utf-8')
        # AESGCM.encrypt returns ciphertext + tag
        ciphertext = self.aesgcm.encrypt(nonce, data, associated_data=None)
        return nonce + ciphertext

    def decrypt(self, data: bytes) -> str:
        """
        Decrypts bytes to plaintext string.
        Expects: nonce (12 bytes) + ciphertext + tag
        """
        nonce = data[:12]
        ciphertext = data[12:]
        plaintext_bytes = self.aesgcm.decrypt(nonce, ciphertext, associated_data=None)
        return plaintext_bytes.decode('utf-8')
