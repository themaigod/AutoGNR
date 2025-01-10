import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# convolution operation
class Conv(nn.Module):

    def __init__(self, n_hid_in, n_hid_out):
        super(Conv, self).__init__()
        self.n_hid_in = n_hid_in
        self.n_hid_out = n_hid_out

    def forward(self, x, adjs, idx):
        return torch.spmm(adjs[idx], x)


class HeterModel(nn.Module):

    def __init__(self, in_dim, n_hid, num_node_types, num_hops, n_adjs, n_classes, n_atten=256, dropout=0):
        super(HeterModel, self).__init__()

        self.num_hops = num_hops
        self.num_node_types = num_node_types
        self.n_adjs = n_adjs
        self.in_dim = in_dim
        self.n_hid = n_hid

        # self.ws = nn.ModuleList()
        # for i in range(num_node_types):
        #     self.ws.append(nn.Linear(in_dim, in_dim))

        self.aggs = nn.ModuleList()
        for i in range(num_hops):
            self.aggs.append(Conv(in_dim, in_dim))

        self.linear = nn.Linear(in_dim, n_hid)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(n_hid, n_classes)
        self.feats_drop = nn.Dropout(dropout) if dropout is not None else lambda x: x

    def forward(self, node_feats, node_types, adjs, anchor_idx, arch):
        # hid = torch.zeros((node_types.size(0), self.in_dim)).to(device)
        # for i in range(self.num_node_types):
        #     idx = (node_types == i)
        #     hid[idx] = self.ws[i](node_feats[idx])

        hid = node_feats
        hid = self.feats_drop(hid)
        hid_seq = []
        hid_seq.append(F.normalize(hid[anchor_idx], p=2, dim=1))  # self loop

        # access to different neighbor hops nodes which node types contains on the combinations ordered by architecture
        for i in range(len(arch)):
            hid_ = self.aggs[i](hid, adjs[i], arch[i])
            hid_seq.append(F.normalize(hid_, p=2, dim=1))

        output = sum(hid_seq) / len(hid_seq)  # aggregate from all the neighbor hops
        output = self.linear(output)
        output = self.act(output)
        output = self.classifier(output)  # classify
        return output


