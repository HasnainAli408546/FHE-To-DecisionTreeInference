# client/tree_traversal.py
import os
import numpy as np

# Load tree structure once
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tree_path = os.path.join(_project_root, "server", "model", "tree_matrices.npy")
_tree = np.load(_tree_path, allow_pickle=True).item()

children_left = _tree["children_left"]
children_right = _tree["children_right"]
leaf_values = _tree["leaf_values"]      # value per node index
classes = _tree["classes"]              # optional, for labels if needed

def plaintext_traverse_from_scores(scores):
    """
    Simulate scikit-learn's tree traversal but using node scores.
    scores[i] ≈ x[feature_i] - threshold_i for node i.
    Returns predicted class (int).
    """
    node = 0
    while True:
        left = children_left[node]
        right = children_right[node]

        # Leaf node: sklearn encodes leaves as children_left[node] == children_right[node]
        if left == right:
            pred_class = leaf_values[node]
            return int(pred_class)

        s = scores[node]
        # if s <= 0 -> x <= threshold -> go left, else right
        if s <= 0:
            node = left
        else:
            node = right
