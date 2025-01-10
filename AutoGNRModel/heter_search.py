import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Conv1(nn.Module):

    def __init__(self, n_hid_in, n_hid_out, bias=False):
        super(Conv1, self).__init__()
        self.n_hid_in = n_hid_in
        self.n_hid_out = n_hid_out
        self.temperature = 0.07

        self.wt = Parameter(torch.FloatTensor(n_hid_in, n_hid_out))

        if bias:
            self.bias = Parameter(torch.FloatTensor(n_hid_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.wt)

    def forward(self, x, weight, adjs, idx, anchor):
        adj = adjs[idx].to_dense()
        anchor_adj = torch.mm(anchor, self.wt)
        anchor_adj = torch.mm(anchor_adj, x.T)  # [batch_node, num_node]

        paddings = torch.ones_like(anchor_adj) * (-2 ** 32 + 1)

        item_att_w = torch.where(adj > 0, anchor_adj, paddings)
        atten = torch.softmax(item_att_w / self.temperature, dim=1)
        output = torch.mm(atten, x)

        if self.bias is not None:
            return weight[idx] * (output + self.bias)
        else:
            return weight[idx] * output

        # return weight[idx] * torch.spmm(adjs[idx], x)


# convolution operation
class Conv(nn.Module):

    def __init__(self):
        super(Conv, self).__init__()

    def forward(self, x, weight, adjs, idx):

        return weight[idx] * torch.spmm(adjs[idx], x)


class Cell(nn.Module):

    def __init__(self, n_hid_prev, n_hid, use_norm=False, use_nl=True):
        super(Cell, self).__init__()

        # self.affine = nn.Linear(n_hid_prev, n_hid)
        self.norm = nn.LayerNorm(n_hid, elementwise_affine=False) if use_norm is True else lambda x: x
        self.use_nl = use_nl

        self.agg = Conv()

    def forward(self, x, weight, adjs, idx):
        # x = self.affine(x)

        # aggregate from different node types combinations
        if idx is not None:
            out = self.agg(x, weight, adjs, idx)

        else:
            out = sum(self.agg(x, weight, adjs, i)for i in range(len(adjs)))

        # normalize
        output = self.norm(out)
        output = F.normalize(output, p=2, dim=1)

        # activation
        if self.use_nl:
            output = F.gelu(output)

        return output


class LayerAtten(nn.Module):

    def __init__(self, in_features, hidden, temperature=0.07):
        super(LayerAtten, self).__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.temperature = temperature
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(self.in_features * 2, self.hidden, bias=False)
        self.linear2 = nn.Linear(self.hidden, 1, bias=False)

    def forward(self, q_vec, k_vec, v_vec):
        bsz, m, d = k_vec.shape
        q_vec = q_vec.view(-1, 1, self.in_features).repeat(1, m, 1)
        xx = torch.cat((q_vec, k_vec), dim=-1)
        xx = self.linear1(xx)
        xx = self.act(xx)
        xx = self.linear2(xx)   # n_anchor, m, 1
        xx = torch.softmax(xx, dim=1)
        v_vec = torch.mul(v_vec, xx)
        v_vec = torch.sum(v_vec, dim=1)
        return v_vec


class HeterModel(nn.Module):

    def __init__(self, in_dim, n_hid, num_node_types, num_hops, n_adjs, n_classes, criterion, n_atten=256):
        super(HeterModel, self).__init__()

        self._criterion = criterion
        self.num_hops = num_hops
        self.num_node_types = num_node_types
        self.n_adjs = n_adjs
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.n_classes = n_classes

        # self.ws = nn.ModuleList()
        # for i in range(num_node_types):
        #     self.ws.append(nn.Linear(in_dim, n_hid))

        # self.affine = nn.Linear(in_dim, n_hid)

        self.lam_seq = (1e-3 * torch.randn(num_hops, n_adjs)).to(device)
        self.lam_seq.requires_grad_(True)

        self.cells = nn.ModuleList()
        for i in range(num_hops):
            self.cells.append(Cell(in_dim, in_dim))

        # self.atten = LayerAtten(in_dim, n_atten)
        self.linear = nn.Linear(in_dim, n_hid)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(n_hid, n_classes)

    # architecture parameters
    def lambdas(self):

        return [self.lam_seq]

    def sample(self, eps):
        if np.random.uniform() < eps:
            idx_seq = torch.randint(low=0, high=self.lam_seq.size(-1), size=self.lam_seq.size()[:-1]).to(device)
        else:
            idx_seq = torch.argmax(F.softmax(self.lam_seq, dim=-1), dim=-1).T.to(device)
        return idx_seq

    # forward process and calculate the loss
    def _loss(self, node_feats, node_types, adjs, idx_seq, anchor_idx, target):
        logits = self(node_feats, node_types, adjs, idx_seq, anchor_idx).to(device)

        return self._criterion(logits, target)

    # copy the model to create a new one
    def new(self):
        model_new = HeterModel(self.in_dim, self.n_hid, self.num_node_types, self.num_hops, self.n_adjs, self.n_classes,
                               self._criterion)
        model_new.to(device)
        for x, y in zip(model_new.lambdas(), self.lambdas()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, node_feats, node_types, adjs, idx_seq, anchor_idx):
        # hid = torch.zeros((node_types.size(0), self.n_hid)).to(device)
        # for i in range(self.num_node_types):
        #     idx = (node_types == i)
        #     hid[idx] = self.ws[i](node_feats[idx])

        # hid = self.affine(node_feats)
        hid = node_feats
        hop_seq = []
        hop_seq.append(F.gelu(F.normalize(hid[anchor_idx], p=2, dim=1)))  # self loop
        anchor = hop_seq[0]

        alpha_seq = F.softmax(self.lam_seq, dim=-1)  # calculate the attention score

        # do the aggregation from different node types combinations
        if idx_seq is not None:
            for i in range(self.num_hops):
                hid_ = self.cells[i](hid, alpha_seq[i], adjs[i], idx_seq[i])
                hop_seq.append(hid_)
        else:
            for i in range(self.num_hops):
                hid_ = self.cells[i](hid, alpha_seq[i], adjs[i], None)
                hop_seq.append(hid_)

        # hop_seq = torch.stack(hop_seq, dim=1)
        # output = self.atten(anchor, hop_seq, hop_seq)
        output = sum(hop_seq) / len(hop_seq)  # aggregation on different neighbor hops
        output = self.linear(output)
        output = self.act(output)
        output = self.classifier(output)  # classify
        return output

    def extract(self):

        return self.sample(0)  # get current the architecture

