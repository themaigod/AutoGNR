import json
import os
import sys
import numpy as np
import pickle
import scipy.sparse as sp
import logging
import argparse
import torch
import torch.nn.functional as F
import time
from sklearn.metrics import f1_score

from math import ceil
from itertools import combinations
from heter_model import HeterModelFeature
from utils import normalize_row, sparse_mx_to_torch_sparse_tensor, arch_2, arch_3, arch_4, arch_5
import heter_model
# import GeneralSetting

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--num_hops', type=int, default=3, help='number of hops to aggregate')
parser.add_argument('--dataset', type=str, default='ogbn_mag', help='dataset')
# parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=300, help='number of epochs for supernet training')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--result', type=str, default="./")
# parser.add_argument('--labels', type=str, default="../data/ACM/labels_5_fold_cross_validation_0.pkl")
parser.add_argument('--labels', type=str, default="../data/ogbn_mag/labels.pkl")
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
# args.seed = GeneralSetting.seed
# args.num_hops = GeneralSetting.num_hops
# args.dataset = GeneralSetting.dataset
# args.n_hid = GeneralSetting.n_hid
# args.labels = GeneralSetting.labels

config = "lr" + str(args.lr) + "_wd" + str(args.wd) + \
         "_h" + str(args.n_hid) + "_hop" + str(args.num_hops) + \
         "_epoch" + str(args.epochs) + "_drop" + str(args.dropout) + "_s" + str(args.seed)

logdir = os.path.join("log/eval", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(logdir, config + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
device = torch.device("cuda:{}".format(args.device))
device = torch.device("cpu")
heter_model.device = device

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    datadir = "../data"
    datafolder = os.path.join(datadir, args.dataset)

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
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.long).to(device)
    valid_idx = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.long).to(device)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.long).to(device)
    test_idx = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.long).to(device)
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.long).to(device)

    n_classes = train_target.max().item() + 1

    #  create adjacency for specified node type

    adjs_name = 'adjs_' + str(args.num_hops) + '_walks.pkl'

    # load walks
    with open(os.path.join(datafolder, "walks.pkl"), "rb") as f:
        walks = pickle.load(f)
        f.close()

    if not os.path.exists(os.path.join(datafolder, adjs_name)):
        adjs = create_adjs_from_walks(walks, args.num_hops, node_types)
        with open(os.path.join(datafolder, adjs_name), "wb") as f:
            pickle.dump(adjs, f)
            f.close()
    else:
        with open(os.path.join(datafolder, adjs_name), "rb") as f:
            adjs = pickle.load(f)
            f.close()

    node_types = torch.from_numpy(node_types).to(device)

    model = HeterModelFeature(node_feats.size(-1), args.n_hid, num_node_types, args.num_hops, len(adjs[0]), n_classes,
                       dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val = None
    final_macro = None
    final_micro = None
    anchor = None
    patience = 0

    # if args.num_hops == 2:
    #     arch = arch_2
    # elif args.num_hops == 3:
    #     arch = arch_3
    # elif args.num_hops == 4:
    #     arch = arch_4
    # elif args.num_hops == 5:
    #     arch = arch_5

    with open("./utils.json", "r") as f:
        arches = json.load(f)
        arch = arches['arch_{}'.format(args.num_hops)]

    adjs_train = get_adjs(adjs, train_idx.cpu(), device)
    adjs_val = get_adjs(adjs, valid_idx.cpu(), device)
    adjs_test = get_adjs(adjs, test_idx.cpu(), device)

    print('Training begin!')
    for epoch in range(args.epochs):
        train_loss = train(node_feats, node_types, adjs_train, train_idx, train_target, model, optimizer, arch)
        val_loss, f1_val_macro, f1_val_micro, f1_test_macro, f1_test_micro = predict(node_feats, node_types, adjs_val,
                                                                                     adjs_test, valid_idx, valid_target,
                                                                                     test_idx,
                                                                                     test_target, model, arch)
        logging.info(
            "Epoch {}; Train err {}; Val err {}; F1_test {}".format(epoch + 1, train_loss, val_loss, f1_test_macro))
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            final_macro = f1_test_macro
            final_micro = f1_test_micro
            anchor = epoch + 1
            patience = 0
            best_feature = model.feature_temp
        else:
            patience += 1
            if patience == 100:
                break

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    X_tsne = TSNE(random_state=args.seed).fit_transform(best_feature.cpu().detach().numpy())
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=test_target.cpu().numpy())
    plt.savefig('tsne.png')

    logging.info("Best val {} at epoch {}; Test score {},{}".format(best_val, anchor, final_micro, final_macro))
    labels = os.path.split(args.labels)[1][-5]
    name = "ours.csv"
    result_path = os.path.join(args.result, name)

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    import csv
    with open(result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.dataset, labels, args.seed, final_micro, final_macro, args.num_hops, args.n_hid, args.num_hops, args.lr])

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


def create_adjs(adj, num_hops, node_types):
    n_node, _ = adj.shape
    node_type = list(set(node_types.tolist()))
    num_node_types = len(node_type)

    adj_selfloop = adj
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape, dtype=np.float32)
    adj_orig = adj  # csr
    adj_hops = []  # csr //n_hop adjacency of all types

    for i in range(num_hops):
        if i == 0:
            temp = adj_orig.T
        else:
            adj = sp.csr_matrix(adj.dot(adj_orig.toarray()))
            temp = adj.T

        print('---> Sparse rate of %d-hops adjacency is : %.4f' % (i + 1, temp.nnz / n_node / n_node))
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
        temp = [sparse_mx_to_torch_sparse_tensor(adj_[anchor_idx].tocoo()).to(device) for adj_ in adjs[i]]
        adjs_anchors.append(temp)

    return adjs_anchors


def train(node_feats, node_types, adjs_train, train_idx, train_target, model, optimizer, arch):
    model.train()
    optimizer.zero_grad()
    out = model(node_feats, node_types, adjs_train, train_idx, arch[args.dataset])
    loss = F.cross_entropy(out, train_target)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(node_feats, node_types, adjs_val, adjs_test, valid_idx, valid_target, test_idx, test_target, model, arch):
    model.eval()
    with torch.no_grad():
        out_val = model(node_feats, node_types, adjs_val, valid_idx, arch[args.dataset])
        out_test = model(node_feats, node_types, adjs_test, test_idx, arch[args.dataset])
    loss = F.cross_entropy(out_val, valid_target)
    f1_val_macro = f1_score(valid_target.cpu().numpy(), torch.argmax(out_val, dim=-1).cpu().numpy(), average='macro')
    f1_test_macro = f1_score(test_target.cpu().numpy(), torch.argmax(out_test, dim=-1).cpu().numpy(), average='macro')
    f1_val_micro = f1_score(valid_target.cpu().numpy(), torch.argmax(out_val, dim=-1).cpu().numpy(), average='micro')
    f1_test_micro = f1_score(test_target.cpu().numpy(), torch.argmax(out_test, dim=-1).cpu().numpy(), average='micro')
    return loss.item(), f1_val_macro, f1_val_micro, f1_test_macro, f1_test_micro


if __name__ == '__main__':
    main()
