import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from shared.config import AES_KEY

aesgcm = AESGCM(AES_KEY)

def decrypt_payload(iv_b64, ct_b64):
    iv = base64.b64decode(iv_b64)
    ct = base64.b64decode(ct_b64)
    return aesgcm.decrypt(iv, ct, None)
