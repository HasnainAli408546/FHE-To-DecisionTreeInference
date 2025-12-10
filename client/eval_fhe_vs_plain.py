# client/eval_fhe_vs_plain.py
import time
import numpy as np

from client.data_utils import load_iris_test_split
from client.plain_predict import plain_predict
from client.client import fhe_predict

def main():
    X_test, y_test = load_iris_test_split()
    n = len(X_test)

    plain_correct = 0
    fhe_correct = 0
    match_plain_fhe = 0
    total_times = []

    for i, (x, y_true) in enumerate(zip(X_test, y_test)):
        x = x.astype(float)

        # Plaintext prediction
        y_plain = plain_predict(x)
        if y_plain == y_true:
            plain_correct += 1

        # FHE prediction + timing
        t0 = time.perf_counter()
        y_fhe = fhe_predict(x)
        t1 = time.perf_counter()
        total_times.append(t1 - t0)

        if y_fhe == y_true:
            fhe_correct += 1
        if y_fhe == y_plain:
            match_plain_fhe += 1

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"Processed {i+1}/{n} samples...")

    print("\n=== Accuracy & Match Metrics ===")
    print(f"Test samples              : {n}")
    print(f"Plaintext accuracy        : {plain_correct / n:.3f}")
    print(f"FHE accuracy              : {fhe_correct / n:.3f}")
    print(f"FHE vs plaintext match    : {match_plain_fhe / n:.3f}")

    total_times = np.array(total_times)
    print("\n=== Timing (total FHE pipeline, client-visible) ===")
    print(f"Mean per-sample time      : {total_times.mean()*1000:.2f} ms")
    print(f"Median per-sample time    : {np.median(total_times)*1000:.2f} ms")
    print(f"Min / Max                 : {total_times.min()*1000:.2f} / {total_times.max()*1000:.2f} ms")

if __name__ == "__main__":
    main()
