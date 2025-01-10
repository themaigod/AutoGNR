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

from itertools import combinations
from heter_search import HeterModel
from utils import normalize_row, sparse_mx_to_torch_sparse_tensor
from architecture import Architect
import heter_search
import architecture
import json

parser = argparse.ArgumentParser()  # some settings to change, include many hyper parameters
parser.add_argument('--eta', type=float, default=0.025, help='eta')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--awd', type=float, default=0, help='weight decay for arch encoding')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--num_hops', type=int, default=3, help='maximum number of neighbor hops to aggregate')
parser.add_argument('--dataset', type=str, default='IMDB', help="which dataset you use")
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs for supernet training')
# parser.add_argument('--eps', type=float, default=0.3, help='probability of random sampling')
parser.add_argument('--decay', type=float, default=0.9, help='decay factor for eps')
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--labels', type=str, default="../data/IMDB/labels_5_fold_cross_validation_4.pkl",
                    help="labels path with split")
parser.add_argument('--device', type=int, default=0, help="the gpu device")
args = parser.parse_args()

# saving the log
config = "lr" + str(args.lr) + "_wd" + str(args.wd) + \
         "_h" + str(args.n_hid) + "_alr" + str(args.alr) + \
         "_hop" + str(args.num_hops) + "_epoch" + str(args.epochs) + \
         "_d" + str(args.decay) + "_s" + str(args.seed)

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
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get datadir and datafolder
    datadir = "../data"
    datafolder = os.path.join(datadir, args.dataset)

    # loading dataset
    print('loading data...')
    #  load data
    with open(os.path.join(datafolder, "node_features.pkl"), "rb") as f:
        node_feats = pickle.load(f)
        f.close()
    node_feats = torch.from_numpy(node_feats.astype(np.float32)).to(device)

    node_types = np.load(os.path.join(datafolder, "node_types.npy"))
    num_node_types = len(set(node_types.tolist()))

    with open(os.path.join(datafolder, "edges.pkl"), "rb") as f:
        edges = pickle.load(f)
        f.close()

    # add self loop and mix all the sub graph
    adj_orig = sum([sub_adj.astype(np.float32) for sub_adj in edges])
    adj_orig += sp.eye(adj_orig.shape[0], dtype=np.float32)
    adj_orig[adj_orig > 0] = 1

    # * load labels
    with open(args.labels, "rb") as f:
        labels = pickle.load(f)
        f.close()

    # get train, valid and test set
    train_idx = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.long).to(device)
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.long).to(device)
    valid_idx = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.long).to(device)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.long).to(device)

    n_classes = train_target.max().item() + 1

    #  create adjacency for specified node types combinations
    adjs_name = 'adjs_' + str(args.num_hops) + '.pkl'

    print('pre-processing data...')
    if not os.path.exists(os.path.join(datafolder, adjs_name)):
        adjs = create_adjs(adj_orig, args.num_hops, node_types)
        with open(os.path.join(datafolder, adjs_name), "wb") as f:
            pickle.dump(adjs, f)
            f.close()
    else:
        with open(os.path.join(datafolder, adjs_name), "rb") as f:
            adjs = pickle.load(f)
            f.close()

    node_types = torch.from_numpy(node_types).to(device)

    # get model, loss function, optimization, architecture object for architecture parameters update
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    model = HeterModel(node_feats.size(-1), args.n_hid, num_node_types, args.num_hops, len(adjs[0]), n_classes,
                       criterion).to(device)

    optimizer_w = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.wd
    )

    architect = Architect(model, args, [0.9, 0.999])

    # eps = args.eps
    # get the graph from the anchor nodes to different neighbor hops on different node types combinations
    adjs_train = get_adjs(adjs, train_idx.cpu(), device)  # get the parts for train
    adjs_val = get_adjs(adjs, valid_idx.cpu(), device)
    best_val = None
    final = None

    print('Training begin!')
    patience = 0
    for epoch in range(args.epochs):
        # train the model and architecture parameters
        train_error, val_error = train(node_feats, node_types, adjs_train, adjs_val, train_idx, train_target, valid_idx,
                                       valid_target, model, optimizer_w, architect, unrolled=1)
        logging.info(
            "Epoch {}; Train err {}; Val err {}; Arch {}".format(epoch + 1, train_error, val_error, model.extract()))

        # eps = eps * args.decay

        if best_val is None or val_error < best_val:  # early stopping with patience 10 by val loss
            best_val = val_error
            final = model.extract()
        else:
            patience += 1
            if patience == 10:
                break
    logging.info(
        "Final Val err {}; Final Arch {}".format(best_val, final))

    # save the architecture
    if os.path.isfile("./utils.json"):
        with open("./utils.json", "r") as f:
            arch = json.load(f)
    else:
        arch = {}

    with open("./utils.json", "w") as f:
        arch.setdefault('arch_{}'.format(args.num_hops), {})
        arch['arch_{}'.format(args.num_hops)][args.dataset] = final.cpu().detach().numpy().tolist()
        json.dump(arch, f)


def create_adjs(adj, num_hops, node_types):
    n_node, _ = adj.shape
    node_type = list(set(node_types.tolist()))
    num_node_types = len(node_type)

    # generate nodes on different neighbor hops
    adj_selfloop = adj
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape, dtype=np.float32)
    adj_orig = adj  # csr
    adj_hops = []  # csr //n_hop adjacency of all types

    for i in range(num_hops):
        if i == 0:
            temp = adj_orig.T
        else:
            adj = sp.csr_matrix(adj.dot(adj_orig))
            temp = adj.T

        print('---> Sparse rate of %d-hops adjacency is : %.4f' % (i + 1, temp.nnz / n_node / n_node))
        adj_hops.append(temp)

    # separate graph on different neighbor hops by node types combinations
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


# get the parts for ordering the index
def get_adjs(adjs, anchor_idx, device):
    adjs_anchors = []
    for i in range(len(adjs)):
        temp = [sparse_mx_to_torch_sparse_tensor(adj_[anchor_idx].tocoo()).to(device) for adj_ in adjs[i]]
        adjs_anchors.append(temp)

    return adjs_anchors


def train(node_feats, node_types, train_adjs, valid_adjs, train_idx, train_target, valid_idx, valid_target, model,
          optimizer_w, architect, unrolled=1):
    # idx_seq = model.sample(eps)
    idx_seq = None
    print(F.softmax(model.lam_seq, dim=-1))
    # print(model.lam_seq)
    # print(list(a for a, b in model.named_parameters()))

    # train the architecture parameters
    architect.step(node_feats, node_types, train_adjs, valid_adjs, idx_seq, train_idx, valid_idx, train_target,
                   valid_target, args.eta, optimizer_w, unrolled)

    # train the model parameters
    k = 1
    for i in range(k):
        optimizer_w.zero_grad()
        # out = model(node_feats, node_types, adjs_train, idxes_seq, train_idx)
        # loss_w = F.cross_entropy(out, train_target)
        loss_w = model._loss(node_feats, node_types, train_adjs, idx_seq, train_idx, train_target)
        loss_w.backward()
        optimizer_w.step()

    loss_a = model._loss(node_feats, node_types, valid_adjs, idx_seq, valid_idx, valid_target)

    return loss_w.item(), loss_a.item()


if __name__ == '__main__':
    main()
