import os, base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from shared.config import AES_KEY

aesgcm = AESGCM(AES_KEY)

def encrypt_payload(data_bytes):
    iv = os.urandom(12)
    ct = aesgcm.encrypt(iv, data_bytes, None)
    return base64.b64encode(iv).decode(), base64.b64encode(ct).decode()
