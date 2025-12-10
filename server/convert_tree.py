import joblib
import numpy as np
import os

# Get current directory (server/) and model path
MODEL_PATH = os.path.join('model', 'dt_plain.joblib')

# Load plaintext trained decision tree model
print("Loading trained decision tree...")
clf = joblib.load(MODEL_PATH)

# Extract tree structure parameters
tree = clf.tree_

# Number of nodes
num_nodes = tree.node_count

# Extract thresholds: -2 means leaf node
thresholds = tree.threshold

# Extract left and right children indices (-1 leaves)
children_left = tree.children_left
children_right = tree.children_right

# Extract feature indices for split
features = tree.feature

# Extract leaf values (value attribute gives class counts)
leaf_values = []
leaf_indices = []
for i in range(num_nodes):
    if children_left[i] == children_right[i]:  # leaf node
        # Extract value vector for classification, take argmax for prediction
        leaf_pred = np.argmax(tree.value[i][0])
        leaf_values.append(leaf_pred)
        leaf_indices.append(i)
    else:
        leaf_values.append(None)

# Convert to numpy arrays for easy matrix form conversion
thresholds = np.array(thresholds)
children_left = np.array(children_left)
children_right = np.array(children_right)
features = np.array(features)
leaf_values = np.array(leaf_values)
leaf_indices = np.array(leaf_indices)

print(f"✅ Tree extraction complete!")
print(f"Number of nodes: {num_nodes}")
print(f"Number of leaf nodes: {len(leaf_indices)}")
print(f"Thresholds shape: {thresholds.shape}")
print(f"Features shape: {features.shape}")
print(f"Children left shape: {children_left.shape}")
print(f"Leaf values shape: {leaf_values.shape}")

# Save extracted tree parameters for later use (Step 3+)
tree_matrices = {
    'thresholds': thresholds,
    'features': features,
    'children_left': children_left,
    'children_right': children_right,
    'leaf_values': leaf_values,
    'leaf_indices': leaf_indices,
    'num_nodes': num_nodes,
    'n_features': clf.n_features_in_,
    'classes': clf.classes_
}

output_path = os.path.join('model', 'tree_matrices.npy')
np.save(output_path, tree_matrices)
print(f"💾 Saved tree matrices to: {output_path}")

# Quick verification
print("\n🔍 Sample data:")
print(f"First 5 thresholds: {thresholds[:5]}")
print(f"First 5 features: {features[:5]}")
print(f"Sample leaf values: {leaf_values[leaf_indices[:3]]}")
