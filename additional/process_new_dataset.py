import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order
from tqdm import trange
import torch
import os


def bfs_sampling(csr_matrix, start_node=None, max_nodes=None):
    """
    Sample a smaller graph using Breadth-First Search (BFS) Sampling.

    Parameters:
    - csr_matrix: scipy.sparse.csr_matrix representing the original graph.
    - start_node: int, the starting node for BFS. If None, a random node is chosen.
    - max_nodes: int, the maximum number of nodes to include in the sample. If None, all nodes reachable from the start_node are included.

    Returns:
    - scipy.sparse.csr_matrix representing the sampled graph.
    """
    num_nodes = csr_matrix.shape[0]

    # If start_node is not specified, choose a random node as the starting point
    if start_node is None:
        start_node = np.random.randint(0, num_nodes)

    # Perform BFS traversal from the start_node
    order = breadth_first_order(csr_matrix, i_start=start_node, directed=False,
                                return_predecessors=False)

    # If max_nodes is specified, limit the BFS traversal to max_nodes
    if max_nodes is not None:
        order = order[:max_nodes]

    # Create the sampled graph
    sampled_graph = csr_matrix[order, :][:, order]

    return sampled_graph


# Example usage
# Assume 'large_graph' is your scipy.sparse.csr_matrix representing the large graph
# For example, to sample a graph starting from a random node including up to 1000 nodes
# smaller_graph = bfs_sampling(large_graph, max_nodes=1000)

# Example usage:
# dataset = OGB_MAG(root='../data/ogbn_mag', preprocess='metapath2vec')  # Placeholder for actual dataset loading
# improved_sampling_result = year_sample_for_ogbnmag_improved(dataset, split_train_val=True, split_ratio=0.8, seed=42)


def get_reconstructed_indices(train_idx, test_idx, transform_dict):
    """
    Maps the original training and testing indices to their new values after transformation.

    Parameters:
    train_idx: Boolean array indicating the papers selected for the training set.
    test_idx: Boolean array indicating the papers selected for the test set.
    transform_dict: Dictionary mapping original paper indices to new, transformed indices.

    Returns:
    train_reconstruction_idx: Array of training indices in the transformed space.
    test_reconstruction_idx: Array of testing indices in the transformed space.
    """
    # Convert boolean arrays to actual indices
    original_train_indices = np.where(train_idx)[0]
    original_test_indices = np.where(test_idx)[0]

    # Map original indices to reconstructed indices
    train_reconstruction_idx = np.array(
        [transform_dict[i] for i in original_train_indices if i in transform_dict])
    test_reconstruction_idx = np.array(
        [transform_dict[i] for i in original_test_indices if i in transform_dict])

    return train_reconstruction_idx, test_reconstruction_idx


def split_train_val_sets(train_idx, label_train, **kwargs):
    """
    Splits the training indices into training and validation sets based on a specified ratio.

    Parameters:
    train_idx: numpy array of training node indices before splitting.
    label_train: numpy array of labels corresponding to the train_idx.
    kwargs: Additional keyword arguments including 'split_ratio' and 'seed' for customizing the split.

    Returns:
    train_train_idx: Indices of the training set after splitting.
    train_val_idx: Indices of the validation set after splitting.
    label_train_train: Labels for the training set after splitting.
    label_train_val: Labels for the validation set after splitting.
    """
    import numpy as np

    # Seed for reproducibility
    seed = kwargs.get("seed", 0)
    np.random.seed(seed)

    # Shuffle indices
    shuffled_indices = np.random.permutation(len(train_idx))

    # Split ratio
    split_ratio = kwargs.get("split_ratio", 0.8)
    split_point = int(len(train_idx) * split_ratio)

    # Splitting
    train_indices = shuffled_indices[:split_point]
    val_indices = shuffled_indices[split_point:]

    # Mapping split indices to actual training indices
    train_train_idx = train_idx[train_indices]
    train_val_idx = train_idx[val_indices]

    # Splitting labels accordingly
    label_train_train = label_train[train_indices]
    label_train_val = label_train[val_indices]

    return train_train_idx, train_val_idx, label_train_train, label_train_val


def update_node_features_for_selected_nodes(node_feats, filtered_node_indices):
    """
    Updates node features to include only those for filtered nodes.

    Parameters:
    node_feats: Dictionary where keys are node types and values are numpy arrays of node features.
    filtered_node_indices: numpy array of node indices that are connected to the selected papers.

    Returns:
    updated_node_feats: numpy array with updated node features for the filtered nodes.
    """
    node_feats = [value for value in node_feats.values()]
    node_feats = np.concatenate(node_feats, axis=0)
    updated_node_feats = node_feats[filtered_node_indices]

    return updated_node_feats


def get_ogbnmag_edges(edge_index_dict, node_feats):
    edges_only = [i.numpy().copy() for i in edge_index_dict.values()]
    num_nodes_each = [len(i) for i in node_feats]
    edges = []
    col_row_temp = edges_only[0]
    col_row_temp[0] = col_row_temp[0] + num_nodes_each[0]
    col_row_temp[1] = col_row_temp[1] + num_nodes_each[0] + num_nodes_each[1]
    edges.append(sp.csr_matrix((np.ones(edges_only[0].shape[1]), col_row_temp),
                               shape=(sum(num_nodes_each), sum(num_nodes_each))))
    col_row_temp = edges_only[1]
    col_row_temp[0] = col_row_temp[0] + num_nodes_each[0]
    edges.append(sp.csr_matrix((np.ones(edges_only[1].shape[1]), col_row_temp),
                               shape=(sum(num_nodes_each), sum(num_nodes_each))))
    col_row_temp = edges_only[2]
    edges.append(sp.csr_matrix((np.ones(edges_only[2].shape[1]), col_row_temp),
                               shape=(sum(num_nodes_each), sum(num_nodes_each))))
    col_row_temp = edges_only[3]
    col_row_temp[1] = col_row_temp[1] + num_nodes_each[0] + num_nodes_each[1] + \
                      num_nodes_each[2]
    edges.append(sp.csr_matrix((np.ones(edges_only[3].shape[1]), col_row_temp),
                               shape=(sum(num_nodes_each), sum(num_nodes_each))))
    return edges


def load_freebase(path):
    """
    Load Freebase dataset from the specified path.

    Parameters:
    path: str, path to the Freebase dataset.

    Returns:
    node_feats: numpy array of node features.
    node_types: numpy array of node types.
    edges: List of scipy.sparse.csr_matrix representing the adjacency matrices.
    labels: List of training, validation, and test sets with their labels.
    """
    node_ids, node_feats, node_types = load_freebase_node(path)
    check_node_ids = np.arange(len(node_ids))
    if not np.all(np.array(node_ids) == check_node_ids):
        raise ValueError("Node IDs are not consistent with the node indices.")
    edges = load_freebase_edge(path, node_nums=len(node_ids), node_types=node_types)
    label_train, label_test, label_num = load_freebase_label(path)
    return node_feats, node_types, edges, label_train, label_test, label_num


def load_freebase_node(path):
    """
    Load Freebase node features and types from the specified path.

    Parameters:
    path: str, path to the Freebase dataset.

    Returns:
    node_feats: numpy array of node features.
    node_types: numpy array of node types.
    """
    node_feats = []
    node_types = []
    node_ids = []
    with open(os.path.join(path, 'node.dat'), 'r', encoding='utf-8') as f:
        for line in f:
            line_content = line.split('\t')
            # node_id, node_name, node_type, node_feature(optional)
            node_id = int(line_content[0])
            node_name = line_content[1]
            node_type = int(line_content[2])
            if len(line_content) > 3:
                node_attr = list(map(float, line_content[3].split(',')))
            else:
                node_attr = []
            node_feats.append(node_attr)
            node_types.append(node_type)
            node_ids.append(node_id)

    # require to sort the node_ids to order the node_feats and node_types
    sorted_node_ids = np.argsort(node_ids)
    node_feats = [node_feats[i] for i in sorted_node_ids]
    node_types = [node_types[i] for i in sorted_node_ids]
    node_ids_reordered = [node_ids[i] for i in sorted_node_ids]
    return node_ids_reordered, node_feats, node_types


def load_freebase_edge(path, node_nums, node_types):
    """
    Load Freebase edges from the specified path.

    Parameters:
    path: str, path to the Freebase dataset.
    node_nums: int, number of nodes in the dataset.
    node_types: numpy array of node types.

    Returns:
    edges: List of scipy.sparse.csr_matrix representing the adjacency matrices.
    """
    edges = []
    edge_types = []
    with open(os.path.join(path, 'link.dat'), 'r', encoding='utf-8') as f:
        for line in f:
            line_content = line.split('\t')
            # edge_source, edge_target, edge_id, edge_weight
            edge_source = int(line_content[0])
            edge_target = int(line_content[1])
            # edge_id = int(line_content[2])
            edge_weight = float(line_content[3])
            edge_type = (int(node_types[edge_source]), int(node_types[edge_target]))
            if edge_type not in edge_types:
                edge_types.append(edge_type)
            edge_type_id = edge_types.index(edge_type)
            edges.append((edge_source, edge_target, edge_weight, edge_type_id))

    # construct the adjacency matrix
    adjs = []
    for i in range(len(edge_types)):
        edge_type = edge_types[i]
        edge_list = [j for j in edges if j[3] == i]
        adj = sp.csr_matrix((np.array([j[2] for j in edge_list]),
                             (np.array([j[0] for j in edge_list]),
                              np.array([j[1] for j in edge_list]))),
                            shape=(node_nums, node_nums))
        adjs.append(adj)

    return adjs


def load_freebase_label(path):
    train_label, label_num = load_freebase_label_named(path, "label.dat")
    test_label, _ = load_freebase_label_named(path, "label.dat.test")
    return train_label, test_label, label_num


def load_freebase_label_named(path, name):
    """
    Load Freebase labels from the specified path.

    Parameters:
    path: str, path to the Freebase dataset.
    name: str, name of the label file.

    Returns:
    labels: List of training, validation, and test sets with their labels.
    label_num: int, number of label classes.
    """
    labels = []
    with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
        for line in f:
            line_content = line.split('\t')
            # node_id, node_name, node_type, node_label
            node_id = int(line_content[0])
            try:
                node_label = int(line_content[3])
            except:
                print(line_content)
            labels.append((node_id, node_label))

    # get the number of classes
    label_classes = set([i[1] for i in labels])
    label_num = len(label_classes)

    return labels, label_num


def separate_train_val(seed, separate_ratio, label_train):
    train_num = int(len(label_train) * separate_ratio)
    import random
    random.seed(seed)
    split_index = random.sample(range(len(label_train)), train_num)
    val_index = [i for i in range(len(label_train)) if i not in split_index]
    label_train_val = [label_train[i] for i in val_index]
    label_train_train = [label_train[i] for i in split_index]
    return label_train_train, label_train_val


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="DBLP2")
    parser.add_argument('--sample_method', type=str, default="None")
    parser.add_argument('--regular_sampling', type=bool, default=True)
    parser.add_argument('--quick', type=bool, default=True)
    args = parser.parse_args()

    if args.dataset == "PubMed":
        if not args.quick:
            path = os.path.join("../data", args.dataset)
            node_feats, node_types, edges, label_train, label_test, label_num = load_freebase(
                os.path.join(path, "ori"))
            # separate the training and validation set
            separate_ratio = 0.8
            label_train_train, label_train_val = separate_train_val(0, separate_ratio,
                                                                    label_train)

            # save the results
            import pickle

            node_feats = np.array(node_feats)
            with open(os.path.join(path, 'node_features.pkl'), 'wb') as f:
                pickle.dump(node_feats, f)

            with open(os.path.join(path, 'node_types.npy'), 'wb') as f:
                np.save(f, node_types)

            edges += [i.transpose(copy=True) for i in edges]
            with open(os.path.join(path, 'edges.pkl'), 'wb') as f:
                pickle.dump(edges, f)

            labels = [label_train_train, label_train_val, label_test]
            with open(os.path.join(path, 'labels.pkl'), 'wb') as f:
                pickle.dump(labels, f)

        else:
            import pickle

            path = os.path.join("../data", args.dataset)
            with open(os.path.join(path, 'node_features.pkl'), 'rb') as f:
                node_feats = pickle.load(f)

            with open(os.path.join(path, 'node_types.npy'), 'rb') as f:
                node_types = np.load(f)

            with open(os.path.join(path, 'edges.pkl'), 'rb') as f:
                edges = pickle.load(f)

            with open(os.path.join(path, 'labels.pkl'), 'rb') as f:
                labels = pickle.load(f)

        import networkx
        from random_walk import RandomWalk

        adj_orig: sp.csr_matrix = sum([sub_adj.astype(np.float32) for sub_adj in edges])
        adj_orig.setdiag(0)
        adj_orig.eliminate_zeros()
        graph = networkx.from_scipy_sparse_matrix(adj_orig)
        rw = RandomWalk(graph, walk_length=4, num_walks=1000, workers=1,
                        temp_folder="../temp/walk")
        graphs = rw.convert_walks_to_graphs()

        pickle.dump(graphs, open(os.path.join(path, 'walks.pkl'), 'wb'))


    elif args.dataset == "DBLP2":
        if not args.quick:
            path = os.path.join("../data", args.dataset)
            node_feats, node_types, edges, label_train, label_test, label_num = load_freebase(
                os.path.join(path, "ori"))
            # separate the training and validation set
            separate_ratio = 0.8
            label_train_train, label_train_val = separate_train_val(0, separate_ratio,
                                                                    label_train)

            # save the results
            import pickle

            node_feats = np.array(node_feats)
            with open(os.path.join(path, 'node_features.pkl'), 'wb') as f:
                pickle.dump(node_feats, f)

            with open(os.path.join(path, 'node_types.npy'), 'wb') as f:
                np.save(f, node_types)

            edges += [i.transpose(copy=True) for i in edges]
            with open(os.path.join(path, 'edges.pkl'), 'wb') as f:
                pickle.dump(edges, f)

            labels = [label_train_train, label_train_val, label_test]
            with open(os.path.join(path, 'labels.pkl'), 'wb') as f:
                pickle.dump(labels, f)

        else:
            import pickle

            path = os.path.join("../data", args.dataset)
            with open(os.path.join(path, 'node_features.pkl'), 'rb') as f:
                node_feats = pickle.load(f)

            with open(os.path.join(path, 'node_types.npy'), 'rb') as f:
                node_types = np.load(f)

            with open(os.path.join(path, 'edges.pkl'), 'rb') as f:
                edges = pickle.load(f)

            with open(os.path.join(path, 'labels.pkl'), 'rb') as f:
                labels = pickle.load(f)

        import networkx
        from random_walk import RandomWalk

        walk_length = 4

        if not os.path.exists(os.path.join(path, 'adj_orig.pkl')):
            adj_orig: sp.csr_matrix = sum(
                [sub_adj.tolil().astype(np.float32) for sub_adj in edges])
            adj_orig.setdiag(0)
            adj_orig.eliminate_zeros()
            with open(os.path.join(path, 'adj_orig.pkl'), 'wb') as f:
                pickle.dump(adj_orig, f)

        else:
            with open(os.path.join(path, 'adj_orig.pkl'), 'rb') as f:
                adj_orig = pickle.load(f)
        # graph = networkx.from_scipy_sparse_matrix(adj_orig)
        labeled_nodes = [i[0] for i in labels[0]] + [i[0] for i in labels[1]] + [i[0]
                                                                                 for i
                                                                                 in
                                                                                 labels[
                                                                                     2]]
        graph = networkx.from_scipy_sparse_array(adj_orig)

        rw = RandomWalk(graph, walk_length=walk_length, num_walks=1000, workers=1,
                        temp_folder="../temp/walk",
                        source_node=labeled_nodes,
                        optp=True, not_pre=True)
        graphs = rw.convert_walks_to_graphs()

        pickle.dump(graphs, open(os.path.join(path, 'walks.pkl'), 'wb'))
