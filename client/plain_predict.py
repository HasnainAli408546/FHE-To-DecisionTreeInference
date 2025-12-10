# client/plain_predict.py
import os
import joblib
import numpy as np

# Load once at import time
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_path = os.path.join(_project_root, "server", "model", "dt_plain.joblib")
_clf = joblib.load(_model_path)

def plain_predict(sample):
    """
    sample: list/array of features WITHOUT bias (e.g. [5.1, 3.5, 1.4, 0.2])
    """
    x = np.array(sample, dtype=float).reshape(1, -1)
    pred = _clf.predict(x)[0]
    return int(pred)
