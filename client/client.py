# client/client.py
import base64
import os
import sys
import time

import requests
import numpy as np
# ------------------------------------------------------------------
# Ensure project root is on sys.path (for shared.config etc.)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports (since this file lives in client/)
from client.fhe_encrypt import (
    create_context_with_secret,
    encrypt_vector_and_serialize,
    serialize_public_context,
)
from client.security import encrypt_payload
from client.load_leaf_outputs import load_leaf_outputs 
from client.tree_traversal import plaintext_traverse_from_scores

SERVER = "http://127.0.0.1:5000/infer"


def generate_nonce() -> str:
    return base64.b64encode(os.urandom(16)).decode()


def build_wrapped_payload(context_bytes: bytes, ciphertext_bytes: bytes) -> bytes:
    """
    Build concatenated payload and then AES-wrap it.

    Format:
      b"TS_CTX::" + base64(context_bytes) + b"::TS_CT::" + base64(ciphertext_bytes)
    Server parses this after AES-GCM decryption.
    """
    ctx_b64 = base64.b64encode(context_bytes)
    ct_b64 = base64.b64encode(ciphertext_bytes)
    payload_bytes = b"TS_CTX::" + ctx_b64 + b"::TS_CT::" + ct_b64
    return payload_bytes


def send_encrypted_request(vector):
    # 1) Create client TenSEAL context (with secret key)
    ctx = create_context_with_secret()

    # 2) Serialize public context (for server)
    public_ctx_bytes = serialize_public_context(ctx)

    # 3) Encrypt input vector and serialize ciphertext
    fhe_ct_bytes = encrypt_vector_and_serialize(ctx, vector)

    # 4) Build payload (concat + AES-GCM wrap)
    payload_bytes = build_wrapped_payload(public_ctx_bytes, fhe_ct_bytes)
    iv_b64, ct_b64 = encrypt_payload(payload_bytes)

    json_data = {
        "nonce": generate_nonce(),
        "timestamp": time.time(),
        "payload": {
            "iv": iv_b64,
            "ct": ct_b64,
        }
        # fhe_context / ciphertext are not sent separately; they are inside payload_bytes
    }

    # 5) Send to server
    r = requests.post(SERVER, json=json_data, timeout=10)
    print("STATUS:", r.status_code)
    try:
        resp = r.json()
    except Exception:
        print("Non-JSON response:", r.text)
        return
    print("SERVER RESPONSE:", resp)

    if r.status_code != 200:
        return

    # 6) Decrypt and deserialize result
    result_b64 = resp.get("result")
    if not result_b64:
        print("no result")
        return

    # Decode base64 → JSON string
    result_bytes = base64.b64decode(result_b64)
    import json
    out = json.loads(result_bytes.decode("utf-8"))

    node_scores_hex = out.get("node_scores", [])
    if not node_scores_hex:
        print("no node_scores in response")
        return

    path_costs_hex = out.get("path_costs", [])
    if not path_costs_hex:
        print("no path_costs in response")
        return
       


    # TenSEAL import (version-compatible loading)
    import tenseal as ts

    # Decrypt first node score as demo
    first_ct_hex = node_scores_hex[0]
    first_ct_bytes = bytes.fromhex(first_ct_hex)
    if hasattr(ts, "ckks_vector_from"):
        first_vec = ts.ckks_vector_from(ctx, first_ct_bytes)
    else:
        first_vec = ts.CKKSVector.load(ctx, first_ct_bytes)
    first_plain = first_vec.decrypt()[0]
    print("DECRYPTED FIRST NODE SCORE (approx):", first_plain)

    # Decrypt all node scores
    all_scores = []
    for h in node_scores_hex:
        ct_b = bytes.fromhex(h)
        if hasattr(ts, "ckks_vector_from"):
            v = ts.ckks_vector_from(ctx, ct_b)
        else:
            v = ts.CKKSVector.load(ctx, ct_b)
        all_scores.append(v.decrypt()[0])
    print("ALL NODE SCORES (approx):", all_scores)

        # Decrypt all path costs
    path_costs = []
    for h in path_costs_hex:
        ct_b = bytes.fromhex(h)
        if hasattr(ts, "ckks_vector_from"):
            v = ts.ckks_vector_from(ctx, ct_b)
        else:
            v = ts.CKKSVector.load(ctx, ct_b)
        path_costs.append(v.decrypt()[0])
    print("PATH COSTS (approx):", path_costs)

    # Select predicted leaf: index of minimum path cost
    best_leaf_idx = int(np.argmin(path_costs))
    print("BEST LEAF INDEX:", best_leaf_idx)

        # Map leaf index → class label using plaintext leaf outputs
    leaf_outputs = load_leaf_outputs()          # shape: (num_leaves,)
    predicted_class = int(leaf_outputs[best_leaf_idx])
    print("PREDICTED CLASS (leaf output):", predicted_class)

def fhe_predict(features):
    """
    features: list without bias term, e.g. [5.1, 3.5, 1.4, 0.2]
    Returns predicted class (int) using FHE pipeline.
    """
    # 1) Add bias and encrypt input
    vector = list(features) + [1.0]

    ctx = create_context_with_secret()
    public_ctx_bytes = serialize_public_context(ctx)
    fhe_ct_bytes = encrypt_vector_and_serialize(ctx, vector)
    payload_bytes = build_wrapped_payload(public_ctx_bytes, fhe_ct_bytes)
    iv_b64, ct_b64 = encrypt_payload(payload_bytes)

    json_data = {
        "nonce": generate_nonce(),
        "timestamp": time.time(),
        "payload": {"iv": iv_b64, "ct": ct_b64},
    }

    # 2) Send request to server
    r = requests.post(SERVER, json=json_data, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Server error: {r.status_code}, {r.text}")

    resp = r.json()
    result_b64 = resp.get("result")
    if not result_b64:
        raise RuntimeError("No result in server response")

    # 3) Decode JSON with node_scores (ignore path_costs for prediction)
    result_bytes = base64.b64decode(result_b64)
    import json
    out = json.loads(result_bytes.decode("utf-8"))

    node_scores_hex = out.get("node_scores", [])
    if not node_scores_hex:
        raise RuntimeError("Missing node_scores in response")

    import tenseal as ts

    # 4) Decrypt all node scores
    scores = []
    for h in node_scores_hex:
        ct_b = bytes.fromhex(h)
        if hasattr(ts, "ckks_vector_from"):
            v = ts.ckks_vector_from(ctx, ct_b)
        else:
            v = ts.CKKSVector.load(ctx, ct_b)
        scores.append(v.decrypt()[0])

    # 5) Traverse tree in plaintext using scores
    predicted_class = plaintext_traverse_from_scores(scores)
    return predicted_class

if __name__ == "__main__":
    # Example: 4 features + bias
    sample = [5.1, 3.5, 1.4, 0.2, 1.0]
    send_encrypted_request(sample)
