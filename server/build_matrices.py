import os
import numpy as np

# build_matrices.py

def build_decision_matrices(tree_data):
    """Convert extracted tree → FHE-ready matrices (as in paper)"""
    thresholds = tree_data['thresholds']
    features = tree_data['features']
    children_left = tree_data['children_left']
    children_right = tree_data['children_right']
    leaf_values = tree_data['leaf_values']
    leaf_indices = tree_data['leaf_indices']

    num_nodes = len(thresholds)
    n_features = tree_data['n_features']

    print(f"🔨 Building matrices for {num_nodes} nodes, {n_features} features...")

    # 1. DECISION MATRIX
    decision_matrix = np.zeros((num_nodes, n_features + 1))

    num_decision_nodes = 0
    for i in range(num_nodes):
        if features[i] != -2:  # decision node
            feat_idx = features[i]
            decision_matrix[i, feat_idx] = 1.0              # x_feat
            decision_matrix[i, n_features] = -thresholds[i]  # -threshold
            num_decision_nodes += 1

    # 2. PATH-COST MATRIX (new)
    # We want: path_costs[leaf] = sum_i path_cost_matrix[leaf, i] * node_score[i]
    num_leaves = len(leaf_indices)
    path_cost_matrix = np.zeros((num_leaves, num_nodes))

    # Precompute parent relationships to recover paths
    parent = {0: -1}  # root has no parent
    is_left_child = {}
    for i in range(num_nodes):
        l = children_left[i]
        r = children_right[i]
        if l != -1:
            parent[l] = i
            is_left_child[l] = True
        if r != -1:
            parent[r] = i
            is_left_child[r] = False

    # For each leaf, walk up to root to find path nodes and directions
    for leaf_row, leaf_idx in enumerate(leaf_indices):
        path_nodes = []        # internal nodes on the path
        path_directions = []   # True if go left at that node, False if right

        current = leaf_idx
        while parent[current] != -1:
            p = parent[current]
            go_left = is_left_child[current]
            path_nodes.append(p)
            path_directions.append(go_left)
            current = p

        # Now path_nodes are from leaf→root; reverse to root→leaf
        path_nodes = path_nodes[::-1]
        path_directions = path_directions[::-1]

        # Fill path_cost_matrix row for this leaf
        for node, go_left in zip(path_nodes, path_directions):
            # node_scores[node] is s_i = x_feat - threshold
            # If path takes left (s_i < 0), use +1 * s_i
            # If path takes right (s_i > 0), use -1 * s_i
            if go_left:
                path_cost_matrix[leaf_row, node] = 1.0
            else:
                path_cost_matrix[leaf_row, node] = -1.0

    # 3. LEAF OUTPUT VECTOR
    leaf_output_vector = np.array([leaf_values[leaf_idx] for leaf_idx in leaf_indices])

    print("✅ Decision matrices built!")
    print(f"   Decision nodes: {num_decision_nodes}/{num_nodes}")
    print(f"   Decision matrix shape: {decision_matrix.shape}")
    print(f"   Path-cost matrix shape: {path_cost_matrix.shape}")
    print(f"   Leaf output vector shape: {leaf_output_vector.shape}")
    print(f"   Sample decision row: {decision_matrix[0]}")
    print(f"   Sample leaf outputs: {leaf_output_vector[:5]}")

    return {
        'decision_matrix': decision_matrix,
        'path_cost_matrix': path_cost_matrix,   # NEW
        'leaf_output_vector': leaf_output_vector,
        'leaf_indices': leaf_indices,
        'n_features': n_features,
        'num_nodes': num_nodes,
    }

def main():
    """Main execution - load tree data and build FHE matrices"""
    print("="*60)
    print("STEP 3: BUILDING FHE MATRICES (Paper's Core Innovation)")
    print("="*60)
    
    # Check if tree data exists
    tree_path = 'model/tree_matrices.npy'
    if not os.path.exists(tree_path):
        print(f"❌ Error: {tree_path} not found!")
        print("   Run 'python convert_tree.py' first!")
        return
    
    # Load extracted tree data
    print(f"📂 Loading tree data from {tree_path}...")
    tree_matrices = np.load(tree_path, allow_pickle=True).item()
    
    # Build FHE matrices
    fhe_matrices = build_decision_matrices(tree_matrices)
    
    # Save FHE-ready matrices
    output_path = 'model/fhe_matrices.npy'
    np.save(output_path, fhe_matrices)
    print(f"\n💾 Saved FHE matrices to: {output_path}")
    
    print("\n🎯 READY FOR HOMOMORPHIC INFERENCE!")
    print("   Next: server/fhe_logic.py will use these matrices")
    print("="*60)

if __name__ == "__main__":
    main()
