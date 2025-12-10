# server/server.py
import base64
import os
import sys
import time

from flask import Flask, jsonify, request

# ------------------------------------------------------------------
# Simple local imports (run from server/ directory)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fhe_logic import evaluate_decision_like
from nonce_cache import is_replay
from security import decrypt_payload

app = Flask(__name__)


@app.route("/infer", methods=["POST"])
def infer():
    """
    Expected JSON from client:

    {
      "nonce": "<base64>",            # unique per request
      "timestamp": <unix_ts>,         # float or int
      "payload": {
         "iv": "<base64>",            # AES-GCM IV
         "ct": "<base64>"             # AES-GCM ciphertext wrapping FHE data
      }
      // NOTE: we do NOT require explicit "fhe_context" / "ciphertext" fields,
      // because both context and ciphertext are inside the AES-wrapped payload
      // as: b"TS_CTX::" + base64(context_bytes) + b"::TS_CT::" + base64(ciphertext_bytes)
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "invalid json"}), 400

    nonce = data.get("nonce")
    timestamp = data.get("timestamp")
    payload = data.get("payload")

    if nonce is None or timestamp is None or payload is None:
        return jsonify({"error": "missing fields"}), 400

    # 1. Timestamp freshness check (optional, relaxed for development)
    try:
        ts_val = float(timestamp)
    except Exception:
        return jsonify({"error": "invalid timestamp"}), 400

    # For stricter security later:
    # if abs(time.time() - ts_val) > 300:
    #     return jsonify({"error": "timestamp outside allowed window"}), 400

    # 2. Replay protection
    if is_replay(nonce):
        return jsonify({"error": "replay detected"}), 403

    # 3. Decrypt AES-GCM payload (authenticity + integrity for FHE bytes)
    decrypted = None
    if isinstance(payload, dict) and "iv" in payload and "ct" in payload:
        try:
            decrypted = decrypt_payload(payload["iv"], payload["ct"])
        except Exception as e:
            return (
                jsonify(
                    {
                        "error": "AES-GCM verification failed",
                        "detail": str(e),
                        "hint": "Ensure client and server share AES_KEY and use the same payload format.",
                    }
                ),
                400,
            )
    else:
        return jsonify({"error": "invalid payload structure"}), 400

    # 4. Extract FHE context and ciphertext from decrypted payload
    # Client format (build_wrapped_payload in client.py):
    #   b"TS_CTX::" + base64(context_bytes) + b"::TS_CT::" + base64(ciphertext_bytes)
    fhe_context_bytes = None
    ciphertext_bytes = None

    try:
        if decrypted and b"TS_CTX::" in decrypted:
            parts = decrypted.split(b"::")
            # Expected: [b"TS_CTX", base64_ctx, b"TS_CT", base64_ct]
            if len(parts) >= 4 and parts[0] == b"TS_CTX" and parts[2] == b"TS_CT":
                fhe_context_bytes = base64.b64decode(parts[1])
                ciphertext_bytes = base64.b64decode(parts[3])
        else:
            # If you ever switch to "decrypted is just ciphertext", you can use this branch:
            ciphertext_bytes = decrypted
    except Exception:
        # Parsing failed; leave bytes as None so we error clearly below
        pass

    if fhe_context_bytes is None:
        return jsonify({"error": "no fhe_context found in AES payload (TS_CTX missing)"}), 400
    if ciphertext_bytes is None:
        return jsonify({"error": "no ciphertext found in AES payload (TS_CT missing)"}), 400

    # 5. Run FHE evaluation (matrix × vector on encrypted data)
    try:
        result_ct_bytes = evaluate_decision_like(fhe_context_bytes, ciphertext_bytes)
    except Exception as e:
        return jsonify({"error": "FHE evaluation error", "detail": str(e)}), 500

    # 6. Return serialized encrypted result as base64 string
    return jsonify({"result": base64.b64encode(result_ct_bytes).decode("utf-8")}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)
