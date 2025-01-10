import os
import sys
import numpy as np
import torch
import pickle as pkl
import scipy.sparse as sp


output_type = {
    "DBLP": 1,
    "ACM": [0, 2, 4],
    "IMDB": [0, 2, 4]
}

arch_2 = {
    "DBLP": [2, 4],
    "ACM": [0, 1],
    "IMDB": [0, 2]
}

arch_3 = {
    "DBLP": [4, 1, 4],
    "ACM": [4, 0, 4],
    "IMDB": [2, 0, 2]
}

arch_4 = {
    "DBLP": [6, 2, 6, 5],
    "ACM": [1, 0, 6],
    "IMDB": [4, 5, 4, 1]
}

arch_5 = {
    "DBLP": [6, 2, 6, 5],
    "ACM": [1, 0, 6],
    "IMDB": [4, 5, 4, 2, 2]
}


def normalize_sym(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_row(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx=sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

