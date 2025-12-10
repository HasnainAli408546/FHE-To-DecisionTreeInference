import time

NONCE_TTL = 60
nonce_store = {}

def is_replay(nonce):
    now = time.time()

    # Clean old nonces
    expired = [n for n, exp in nonce_store.items() if exp < now]
    for n in expired:
        del nonce_store[n]

    if nonce in nonce_store:
        return True

    nonce_store[nonce] = now + NONCE_TTL
    return False
