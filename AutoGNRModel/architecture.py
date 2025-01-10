import torch
import numpy as np
import torch.nn as nn
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs]).to(device)


class Architect(object):

    def __init__(self, model, args, network_optimizer_betas):
        self.network_beta1, self.network_beta2 = network_optimizer_betas
        self.network_weight_decay = args.wd
        self.model = model
        self.eps = 1e-8
        self.optimizer = torch.optim.Adam(self.model.lambdas(),
                                          lr=args.alr, betas=(0.9, 0.999),
                                          weight_decay=0)  # the optimizer for architecture parameters

    def step(self, node_feats, node_types, train_adjs, valid_adjs, idx_seq, train_idx, valid_idx, train_target,
             valid_target, eta, network_optimizer, unrolled):
        # backward for architecture parameters
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(node_feats, node_types, train_adjs, valid_adjs, idx_seq, train_idx, valid_idx,
                                         train_target, valid_target, eta, network_optimizer)  # default, second order approximation
        else:
            self._backward_step(node_feats, node_types, valid_adjs, idx_seq, valid_idx, valid_target)  # first order approximation
        # update architecture parameters
        self.optimizer.step()

    # first order approximation
    def _backward_step(self, node_feats, node_types, adjs, idx_seq, valid_idx, valid_target):
        loss = self.model._loss(node_feats, node_types, adjs, idx_seq, valid_idx, valid_target)
        loss.backward()

    # default, second order approximation, the implement is following the math details as paper
    def _backward_step_unrolled(self, node_feats, node_types, train_adjs, valid_adjs, idx_seq, train_idx, valid_idx,
                                train_target, valid_target, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(node_feats, node_types, train_adjs, idx_seq, train_idx, train_target,
                                                      eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(node_feats, node_types, valid_adjs, idx_seq, valid_idx, valid_target)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.lambdas()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, node_feats, node_types, train_adjs, idx_seq, train_idx, train_target)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.lambdas(), dalpha):
            if v.grad is None:
                v.grad = torch.zeros_like(g.data, requires_grad=True).to(device)
                v.grad.data.copy_(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(self, node_feats, node_types, adjs, idx_seq, anchor_idx, target, eta, network_optimizer):
        loss = self.model._loss(node_feats, node_types, adjs, idx_seq, anchor_idx, target)
        theta = _concat(self.model.parameters()).data
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta

        try:
            step = network_optimizer.state[list(self.model.named_parameters())[0][0]]['step']
            exp_avg = _concat(network_optimizer.state[v]['exp_avg'] for v in self.model.parameters())
            exp_avg_sq = _concat(network_optimizer.state[v]['exp_avg_sq'] for v in self.model.parameters())
        except:
            step = 1
            exp_avg = torch.zeros_like(theta).to(device)
            exp_avg_sq = torch.zeros_like(theta).to(device)

        exp_avg.mul_(self.network_beta1).add_(dtheta, alpha=1 - self.network_beta1)
        exp_avg_sq.mul_(self.network_beta2).addcmul_(dtheta, dtheta, value=1 - self.network_beta2)

        bias_correction1 = 1 - self.network_beta1 ** step
        bias_correction2 = 1 - self.network_beta2 ** step
        step_size = eta / bias_correction1

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
        theta.addcdiv_(exp_avg, denom, value=-step_size)

        unrolled_model = self._construct_model_from_theta(theta)
        return unrolled_model

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to(device)

    def _hessian_vector_product(self, vector, node_feats, node_types, adjs, idx_seq, anchor_idx, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(node_feats, node_types, adjs, idx_seq, anchor_idx, target)
        grads_p = torch.autograd.grad(loss, self.model.lambdas())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(node_feats, node_types, adjs, idx_seq, anchor_idx, target)
        grads_n = torch.autograd.grad(loss, self.model.lambdas())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

