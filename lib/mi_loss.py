import math

import torch
import torch.nn.functional as F
from lib.functions import *

""" Compute the disimilarity between a & b"""
def get_loss(a, b, measure):
    if len(a.shape) == len(b.shape) == 3:
        return multi_channel_loss(a, b, measure)
    elif len(a.shape) == len(b.shape) == 2:
        return vector_loss(a, b, measure)
    elif len(a.shape) == len(b.shape) == 1:
        return scalar_loss(a, b, measure)
    else:
        raise NotImplementedError("Wrong Dimension of Input at Loss Computation!")

def multi_channel_loss(l, m, measure):
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    # First we make the input tensors the right shape.
    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device)
    n_mask = 1 - mask

    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False).mean(2).mean(2)
    E_neg = get_negative_expectation(u, measure, average=False).mean(2).mean(2)

    # Mask positive and negative terms for positive and negative parts of loss
    E_pos = (E_pos * mask).sum() / mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos

    return loss

def vector_loss(x, z, measure):
    N, mi_units = x.size()[:2]
    x = x.view(N, mi_units, -1)
    z = z.view(N, mi_units, -1)
    return multi_channel_loss(x, z, measure)

# NOTE: not properly implemented yet
def scalar_loss(z, y, measure):
    N = z.size()[0]
    y = y.view(N, 1, -1)
    z = z.view(N, 1, -1)
    return multi_channel_loss(y, z, measure)

""" loss function = celoss - alpha * zxloss + beta * zyloss """
def total_loss(criterion, predict, target, x_encoded, zx, zy, yc, measure, alpha, beta):
    return criterion(predict, target) - alpha * vector_loss(x_encoded, zx, measure) + beta * scalar_loss(zy, yc, measure)
