# shared/config.py
import os

# AES key (32 bytes for AES-256).
#
# NOTE:
# -----
# Originally this project allowed overriding the AES key via the SFI_AES_KEY
# environment variable. If the client and server were started from different
# environments (or the variable was set incorrectly in one of them), their
# AES keys would not match and AES-GCM decryption would always fail with
# an "AES-GCM verification failed" error.
#
# To make the demo more robust and avoid those errors, we now use a single
# deterministic key for both client and server. As long as both sides import
# this module, they will share the same AES key and AES-GCM decryption will
# succeed for valid requests.
#
# If you want to switch back to environment-based keys for production,
# replace AES_KEY below with a securely generated 32‑byte value loaded from
# a KMS, .env file, or another secure configuration mechanism.
AES_KEY = b'\x01' * 32  # 32 bytes = 256-bit AES key

# FHE (TenSEAL) parameters: keep deterministic values so client & server
# can recreate compatible contexts.
FHE_PARAMS = {
    "poly_modulus_degree": 8192,
    "coeff_mod_bit_sizes": [60, 40, 40, 60],
    "global_scale": 2**40,
}