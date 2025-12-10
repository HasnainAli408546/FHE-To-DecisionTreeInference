# client/data_utils.py
import os
import numpy as np

def load_iris_test_split():
    """
    Load X_test, y_test saved in server/model.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, "server", "model")
    X_test = np.load(os.path.join(model_dir, "X_test.npy"))
    y_test = np.load(os.path.join(model_dir, "y_test.npy"))
    return X_test, y_test
