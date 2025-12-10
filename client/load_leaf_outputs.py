# client/load_leaf_outputs.py
import os
import numpy as np

def load_leaf_outputs():
    """
    Load leaf_output_vector from model/fhe_matrices.npy.
    This is plaintext model data (class per leaf).
    """
    # project_root/ model / fhe_matrices.npy
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "server", "model", "fhe_matrices.npy")

    mats = np.load(path, allow_pickle=True).item()
    return mats["leaf_output_vector"]

