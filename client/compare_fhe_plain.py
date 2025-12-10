# client/compare_fhe_plain.py
from client.plain_predict import plain_predict
from client.client import fhe_predict

def main():
    # Example sample
    sample = [5.1, 3.5, 1.4, 0.2]

    plain = plain_predict(sample)
    fhe = fhe_predict(sample)

    print("Sample:", sample)
    print("Plaintext predicted class:", plain)
    print("FHE predicted class      :", fhe)
    print("MATCH:" , plain == fhe)

if __name__ == "__main__":
    main()
