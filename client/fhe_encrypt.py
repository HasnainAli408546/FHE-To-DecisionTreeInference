# client/fhe_encrypt.py
import tenseal as ts
from shared.config import FHE_PARAMS

def create_context_with_secret():
    """
    Create TenSEAL context with secret key (client-side).
    Returns context object.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=FHE_PARAMS["poly_modulus_degree"],
        coeff_mod_bit_sizes=FHE_PARAMS["coeff_mod_bit_sizes"]
    )
    ctx.global_scale = FHE_PARAMS["global_scale"]
    # generate keys needed for vector operations
    ctx.generate_relin_keys()
    ctx.generate_galois_keys()
    # secret key is present in this context (client must keep it)
    return ctx

def serialize_public_context(ctx) -> bytes:
    """
    Serialize context without secret key so it can be loaded on server.
    """
    return ctx.serialize(save_secret_key=False)

def encrypt_vector_and_serialize(ctx, vector):
    """
    Encrypt python list 'vector' into CKKS vector and return serialized bytes.
    """
    v = ts.ckks_vector(ctx, vector)
    return v.serialize()