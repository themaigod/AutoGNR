import os
import sys
import numpy as np
import pickle
import scipy.sparse as sp
import logging
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

from math import ceil
from itertools import combinations
from heter_search import HeterModel
from utils import normalize_row, sparse_mx_to_torch_sparse_tensor
from architecture import Architect
import heter_search
import architecture
import json

parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, default=0.025, help='eta')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--awd', type=float, default=0,
                    help='weight decay for arch encoding')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4,
                    help='learning rate for architecture parameters')
parser.add_argument('--num_hops', type=int, default=3,
                    help='number of hops to aggregate')
# parser.add_argument('--dataset', type=str, default='IMDB')
parser.add_argument('--dataset', type=str, default='Freebase')
# parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs for supernet training')
parser.add_argument('--eps', type=float, default=0.3,
                    help='probability of random sampling')
parser.add_argument('--decay', type=float, default=0.9, help='decay factor for eps')
# parser.add_argument('--num_samples', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--labels', type=str,
                    # default="../data/IMDB/labels_5_fold_cross_validation_0.pkl")
                    default="../data/Freebase/labels.pkl")
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
# args.seed = GeneralSetting.seed
# args.num_hops = GeneralSetting.num_hops
# args.dataset = GeneralSetting.dataset
# args.n_hid = GeneralSetting.n_hid
# args.labels = GeneralSetting.labels

config = "lr" + str(args.lr) + "_wd" + str(args.wd) + \
         "_h" + str(args.n_hid) + "_alr" + str(args.alr) + \
         "_hop" + str(args.num_hops) + "_epoch" + str(args.epochs) + \
         "_eps" + str(args.eps) + "_d" + str(args.decay) + "_s" + str(args.seed)

logdir = os.path.join("log/search", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(logdir, config + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
device = torch.device("cuda:{}".format(args.device))
architecture.device = device
heter_search.device = device


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    datadir = "../data"
    datafolder = os.path.join(datadir, args.dataset)

    print('loading data...')
    #  load data
    with open(os.path.join(datafolder, "node_features.pkl"), "rb") as f:
        node_feats = pickle.load(f)
        f.close()
    if isinstance(node_feats, list):
        node_feats = np.array(node_feats)
    node_feats = torch.from_numpy(node_feats.astype(np.float32)).to(device)

    node_types = np.load(os.path.join(datafolder, "node_types.npy"))
    num_node_types = len(set(node_types.tolist()))

    # with open(os.path.join(datafolder, "edges.pkl"), "rb") as f:
    #     edges = pickle.load(f)
    #     f.close()
    #
    # adj_orig = sum([sub_adj.astype(np.float32) for sub_adj in edges])
    # adj_orig += sp.eye(adj_orig.shape[0], dtype=np.float32)
    # adj_orig[adj_orig > 0] = 1

    # * load labels
    with open(args.labels, "rb") as f:
        labels = pickle.load(f)
        f.close()

    train_idx = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.long).to(device)
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.long).to(
        device)
    valid_idx = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.long).to(device)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.long).to(
        device)

    n_classes = train_target.max().item() + 1

    #  create adjacency for specified node type

    adjs_name = ('adjs_' + str(args.num_hops) + "_walks.pkl")

    print('pre-processing data...')

    # * load walks
    with open(os.path.join(datafolder, "walks.pkl"), "rb") as f:
        walks = pickle.load(f)
        f.close()

    if not os.path.exists(os.path.join(datafolder, adjs_name)):
        adjs = create_adjs_from_walks_efficient(walks, args.num_hops, node_types)
        with open(os.path.join(datafolder, adjs_name), "wb") as f:
            pickle.dump(adjs, f)
            f.close()
    else:
        with open(os.path.join(datafolder, adjs_name), "rb") as f:
            adjs = pickle.load(f)
            f.close()

    node_types = torch.from_numpy(node_types).to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    model = HeterModel(node_feats.size(-1), args.n_hid, num_node_types, args.num_hops,
                       len(adjs[0]), n_classes,
                       criterion).to(device)

    optimizer_w = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.wd
    )

    architect = Architect(model, args, [0.9, 0.999])

    eps = args.eps
    adjs_train = get_adjs(adjs, train_idx.cpu(), device)
    adjs_val = get_adjs(adjs, valid_idx.cpu(), device)
    best_val = None
    final = None

    print('Training begin!')
    patience = 0
    for epoch in range(args.epochs):
        train_error, val_error = train(node_feats, node_types, adjs_train, adjs_val,
                                       train_idx, train_target, valid_idx,
                                       valid_target, model, optimizer_w, architect, eps,
                                       unrolled=1)
        logging.info(
            "Epoch {}; Train err {}; Val err {}; Arch {}".format(epoch + 1, train_error,
                                                                 val_error,
                                                                 model.extract()))

        eps = eps * args.decay

        if best_val is None or val_error < best_val:
            best_val = val_error
            final = model.extract()
            patience = 0
        else:
            patience += 1
            if patience == 100:
                break
    logging.info(
        "Final Val err {}; Final Arch {}".format(best_val, final))

    if os.path.isfile("./utils.json"):
        with open("./utils.json", "r") as f:
            arch = json.load(f)
    else:
        arch = {}

    with open("./utils.json", "w") as f:
        arch.setdefault('arch_{}'.format(args.num_hops), {})
        arch['arch_{}'.format(args.num_hops)][
            args.dataset] = final.cpu().detach().numpy().tolist()
        json.dump(arch, f)


def create_adjs_from_walks(adj_walks, num_hops, node_types):
    if len(adj_walks) == 0:
        raise ValueError("adj_walks is empty")
    n_node, _ = adj_walks[0].shape
    node_type = list(set(node_types.tolist()))
    num_node_types = len(node_type)

    adj_selfloops = adj_walks[0]
    adjs = [adj_walk.copy() for adj_walk in adj_walks]  # n_hop adjacency of all types
    # remove self loops
    for i in range(num_hops):
        adjs[i] = adjs[i] - sp.diags(adjs[i].diagonal(), 0)

    comb_ls = []
    idx_ls = [i for i in range(num_node_types)]
    adjs_by_type = []
    for ii in range(1, num_node_types + 1):
        comb_ls.extend(list(combinations(idx_ls, ii)))
        # comb_ls.extend(list(combinations(idx_ls,1)))
    for i in range(num_hops):
        temp = []
        temp.append(sp.csr_matrix(np.zeros(adj_selfloops.shape, dtype=np.float32)))
        for comb in comb_ls:
            idx = torch.zeros(node_types.size)
            for t in comb:
                idx += (node_types == t)
            idx = (idx == 0)

            adj = adjs[i].toarray()
            adj[:, idx] = 0
            adj = normalize_row(sp.csr_matrix(adj, dtype=np.float32))
            temp.append(adj.tocsr())
        adjs_by_type.append(temp)

    return adjs_by_type


def create_adjs_from_walks_efficient(adj_walks, num_hops, node_types):
    if not adj_walks:
        raise ValueError("adj_walks is empty")
    n_node, _ = adj_walks[0].shape
    node_types_set = list(set(node_types.tolist()))
    num_node_types = len(node_types_set)

    adjs = [adj_walk.copy() for adj_walk in adj_walks]  # Copy for mutability
    # Efficient self-loop removal
    for i in range(num_hops):
        adjs[i].setdiag(0)
        adjs[i].eliminate_zeros()

    comb_ls = [comb for i in range(1, num_node_types + 1) for comb in
               combinations(range(num_node_types), i)]

    from tqdm import tqdm
    total_iterations = num_hops * len(comb_ls)
    progress_bar = tqdm(total=total_iterations, desc="Initializing")

    adjs_by_type = []
    for i in range(num_hops):
        temp = [sp.csr_matrix((n_node, n_node),
                              dtype=np.float32)]  # Start with an empty sparse matrix for each hop
        for j, comb in enumerate(comb_ls):

            progress_bar.set_description(
                f"Hop {i + 1}/{num_hops}, Comb. {j + 1}/{len(comb_ls)}")

            mask = torch.zeros(node_types.shape[0], dtype=torch.bool)
            for t in comb:
                mask |= (node_types == node_types_set[t])

            adj_masked = adjs[i].multiply(
                mask.numpy()[None, :])  # Use broadcasting for row-wise multiplication
            adj_normalized = normalize_row(adj_masked)
            temp.append(adj_normalized)

            progress_bar.update(1)

        adjs_by_type.append(temp)

    progress_bar.close()

    return adjs_by_type


def create_adjs(adj, num_hops, node_types):
    n_node, _ = adj.shape
    node_type = list(set(node_types.tolist()))
    num_node_types = len(node_type)

    adj_selfloop = adj
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape,
                              dtype=np.float32)
    adj_orig = adj  # csr
    adj_hops = []  # csr //n_hop adjacency of all types

    for i in range(num_hops):
        if i == 0:
            temp = adj_orig.T
        else:
            adj = sp.csr_matrix(adj.dot(adj_orig))
            temp = adj.T

        print('---> Sparse rate of %d-hops adjacency is : %.4f' % (
            i + 1, temp.nnz / n_node / n_node))
        adj_hops.append(temp)

    adjs = []
    comb_ls = []
    idx_ls = [i for i in range(num_node_types)]
    for ii in range(1, num_node_types + 1):
        comb_ls.extend(list(combinations(idx_ls, ii)))
    # comb_ls.extend(list(combinations(idx_ls,1)))
    for i in range(num_hops):
        temp = []
        temp.append(sp.csr_matrix(np.zeros(adj_selfloop.shape, dtype=np.float32)))
        for comb in comb_ls:
            idx = torch.zeros(node_types.size)
            for t in comb:
                idx += (node_types == t)
            idx = (idx == 0)

            adj = adj_hops[i].toarray()
            adj[:, idx] = 0
            adj = normalize_row(sp.csr_matrix(adj, dtype=np.float32))
            temp.append(adj.tocsr())
        adjs.append(temp)

    return adjs


def get_adjs(adjs, anchor_idx, device):
    adjs_anchors = []
    for i in range(len(adjs)):
        temp = [sparse_mx_to_torch_sparse_tensor(adj_[anchor_idx].tocoo()).to(device)
                for adj_ in adjs[i]]
        adjs_anchors.append(temp)

    return adjs_anchors


def train(node_feats, node_types, train_adjs, valid_adjs, train_idx, train_target,
          valid_idx, valid_target, model,
          optimizer_w, architect, eps, unrolled=1
          ):
    # idx_seq = model.sample(eps)
    idx_seq = None
    print(F.softmax(model.lam_seq, dim=-1))
    # print(model.lam_seq)
    # print(list(a for a, b in model.named_parameters()))

    architect.step(node_feats, node_types, train_adjs, valid_adjs, idx_seq, train_idx,
                   valid_idx, train_target,
                   valid_target, args.eta, optimizer_w, unrolled)

    k = 1
    for i in range(k):
        optimizer_w.zero_grad()
        # out = model(node_feats, node_types, adjs_train, idxes_seq, train_idx)
        # loss_w = F.cross_entropy(out, train_target)
        loss_w = model._loss(node_feats, node_types, train_adjs, idx_seq, train_idx,
                             train_target)
        loss_w.backward()
        optimizer_w.step()

    loss_a = model._loss(node_feats, node_types, valid_adjs, idx_seq, valid_idx,
                         valid_target)

    return loss_w.item(), loss_a.item()


if __name__ == '__main__':
    main()
