# server/fhe_logic.py
"""
Homomorphic evaluation logic for decision-tree inference using matrix multiplication.

Aligned with:
- tree_matrices.npy
- fhe_matrices.npy (from build_matrices.py)
- Paper idea: node comparisons as matrix-vector products.
"""

import os
import json
import numpy as np
import tenseal as ts

# ---------------------------------------------------------------------
# Load FHE decision-tree matrices once
# ---------------------------------------------------------------------

_FHE_MATRICES_PATH = os.path.join("model", "fhe_matrices.npy")
if not os.path.exists(_FHE_MATRICES_PATH):
    raise FileNotFoundError(
        f"Missing {_FHE_MATRICES_PATH}. "
        "Run convert_tree.py and build_matrices.py first."
    )

_fhe_mats = np.load(_FHE_MATRICES_PATH, allow_pickle=True).item()
_DECISION_MATRIX = _fhe_mats["decision_matrix"]        # (num_nodes, n_features+1)
_LEAF_OUTPUT_VECTOR = _fhe_mats["leaf_output_vector"]  # (num_leaves,)
_PATH_COST_MATRIX = _fhe_mats["path_cost_matrix"]      # (num_leaves, num_nodes)  <-- NEW
_N_FEATURES = int(_fhe_mats["n_features"])             # without bias

# ---------------------------------------------------------------------
# Core functions (used by server.py)
# ---------------------------------------------------------------------

def deserialize_context(context_bytes: bytes):
    """Load a TenSEAL context from serialized bytes."""
    if not context_bytes:
        raise ValueError("Empty TenSEAL context bytes.")
    return ts.context_from(context_bytes)


def _deserialize_ckks_vector(ctx, ct_bytes: bytes):
    """Deserialize a CKKSVector from bytes, supporting different TenSEAL versions."""
    # Newer-style helper, if present
    if hasattr(ts, "ckks_vector_from"):
        try:
            return ts.ckks_vector_from(ctx, ct_bytes)
        except Exception:
            pass

    # Fallback to older API
    try:
        return ts.CKKSVector.load(ctx, ct_bytes)
    except Exception:
        pass

    raise ValueError("Could not deserialize CKKSVector from provided bytes.")


def evaluate_decision_like(context_bytes: bytes, ct_bytes: bytes) -> bytes:
    """
    Server entry point.

    Args:
        context_bytes: serialized TenSEAL context (public).
        ct_bytes: serialized CKKSVector encoding [x_0, ..., x_{d-1}, 1.0].

    Returns:
        JSON bytes:
        {
          "node_scores": [hex(serialized_score_0), ...],
          "path_costs":  [hex(serialized_cost_leaf0), ...]   # NEW
        }
    """
    if not context_bytes:
        raise ValueError("Missing TenSEAL context bytes; client must send serialized context.")
    if not ct_bytes:
        raise ValueError("Missing ciphertext bytes.")

    # 1) Load context
    ctx = deserialize_context(context_bytes)

    # 2) Load encrypted input vector
    enc_input = _deserialize_ckks_vector(ctx, ct_bytes)

    # 3) Homomorphic matrix-vector multiplication over decision matrix rows
    #    Compute encrypted node scores s_i = <row_i, x_padded>
    enc_scores = []
    for i in range(_DECISION_MATRIX.shape[0]):
        row = _DECISION_MATRIX[i, :].tolist()
        score_i = enc_input.dot(row)     # homomorphic inner product
        enc_scores.append(score_i)

    # 4) Serialize each node score separately
    serialized_scores = [s.serialize() for s in enc_scores]
    node_scores_hex = [b.hex() for b in serialized_scores]

    # 5) Homomorphic path-cost computation (leaf pruning)  <-- NEW
    #    We conceptually want: path_costs = PATH_COST_MATRIX @ node_scores
    #    Each leaf ℓ gets: cost_ℓ = sum_j PATH_COST_MATRIX[ℓ,j] * s_j
    enc_path_costs = []
    num_leaves = _PATH_COST_MATRIX.shape[0]

    for leaf_idx in range(num_leaves):
        row = _PATH_COST_MATRIX[leaf_idx, :].tolist()
        # Build a linear combination Σ_j w_j * s_j
        # Because each s_j is itself a CKKS scalar vector, we:
        #  - scale each s_j by w_j (w_j ∈ {-1, 0, 1} in our construction)
        #  - sum them up.
        acc = None
        for node_idx, w in enumerate(row):
            if w == 0.0:
                continue
            term = enc_scores[node_idx] * w  # scalar multiply
            if acc is None:
                acc = term
            else:
                acc += term
        # If a leaf has no path nodes (should not happen), set cost 0
        if acc is None:
            acc = enc_scores[0] * 0.0
        enc_path_costs.append(acc)

    serialized_costs = [c.serialize() for c in enc_path_costs]
    path_costs_hex = [b.hex() for b in serialized_costs]

    # 6) Return both node_scores and path_costs (for debugging and flexibility)
    return json.dumps(
        {
            "node_scores": node_scores_hex,
            "path_costs": path_costs_hex,
        }
    ).encode("utf-8")

# ---------------------------------------------------------------------
# Local test when running `python fhe_logic.py`
# ---------------------------------------------------------------------

def _create_ckks_context():
    """
    Create a CKKS context compatible with this TenSEAL version.

    Older TenSEAL typically uses:
        ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, coeff_mod_bit_sizes)
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,        # CKKS scheme selector for older versions
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2**40
    return ctx


def _local_plaintext_test():
    """
    Sanity check:
    - Build a local TenSEAL context
    - Encrypt a dummy input
    - Run evaluate_decision_like
    - Decrypt first node score and first path cost and print them
    """
    print("=" * 60)
    print("LOCAL TEST: Homomorphic matrix × vector with path costs")
    print("=" * 60)

    ctx = _create_ckks_context()

    # Example input: 4 features + bias 1.0
    x_plain = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
    x_padded = np.append(x_plain, 1.0)
    print(f"Plain input (padded): {x_padded}")

    enc_x = ts.ckks_vector(ctx, x_padded.tolist())

    ctx_bytes = ctx.serialize()
    ct_bytes = enc_x.serialize()

    out_bytes = evaluate_decision_like(ctx_bytes, ct_bytes)
    out_json = json.loads(out_bytes.decode("utf-8"))

    # First node score
    first_score_hex = out_json["node_scores"][0]
    first_score_ct = bytes.fromhex(first_score_hex)
    if hasattr(ts, "ckks_vector_from"):
        first_score_vec = ts.ckks_vector_from(ctx, first_score_ct)
    else:
        first_score_vec = ts.CKKSVector.load(ctx, first_score_ct)
    first_score_plain = first_score_vec.decrypt()[0]

    # First path cost
    first_cost_hex = out_json["path_costs"][0]
    first_cost_ct = bytes.fromhex(first_cost_hex)
    if hasattr(ts, "ckks_vector_from"):
        first_cost_vec = ts.ckks_vector_from(ctx, first_cost_ct)
    else:
        first_cost_vec = ts.CKKSVector.load(ctx, first_cost_ct)
    first_cost_plain = first_cost_vec.decrypt()[0]

    print(f"First decision row:  {_DECISION_MATRIX[0]}")
    print(f"Decrypted first node score: {first_score_plain}")
    print(f"Decrypted first path cost : {first_cost_plain}")
    print("=" * 60)


if __name__ == "__main__":
    _local_plaintext_test()
