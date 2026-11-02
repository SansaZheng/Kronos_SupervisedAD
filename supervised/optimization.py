"""Controlled imbalance objectives and conservative pretrained fine tuning."""
import math

import numpy as np
import torch
from torch.nn import functional as F


def positive_class_weight(labels, policy='auto', cap=None):
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    if not n0 or not n1:
        raise ValueError('class weighting needs both training classes')
    if policy == 'auto':
        weight = n0 / n1
    elif policy == 'sqrt':
        weight = math.sqrt(n0 / n1)
    elif policy == 'none':
        weight = 1.
    else:
        weight = float(policy)
    if not np.isfinite(weight) or weight <= 0 or (cap is not None and (not np.isfinite(cap) or cap <= 0)):
        raise ValueError('positive class weight and cap must be finite and positive')
    return float(min(weight, cap) if cap is not None else weight)


def labelled_loss(logits, labels, positive_weight, loss_type='bce', focal_gamma=2.):
    if loss_type not in ('bce', 'focal') or not math.isfinite(focal_gamma) or focal_gamma < 0:
        raise ValueError('invalid supervised loss or focal gamma')
    mask = labels >= 0
    if not mask.any():
        return None
    z, y = logits[mask], labels[mask]
    losses = F.binary_cross_entropy_with_logits(z, y, pos_weight=positive_weight, reduction='none')
    if loss_type == 'focal':
        p = torch.sigmoid(z)
        pt = torch.where(y == 1, p, 1 - p)
        losses = losses * (1 - pt).pow(focal_gamma)
    return losses.mean()


def make_optimizer(model, lr, weight_decay, backbone_lr_ratio=.1, layer_decay=1.):
    if not 0 < backbone_lr_ratio <= 1 or not 0 < layer_decay <= 1 or weight_decay < 0:
        raise ValueError('LR ratios must lie in (0,1], and weight decay must be nonnegative')
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role, layer_factor = 'head_adapter', 1.
        if name.startswith('backbone.'):
            role = 'backbone'
            depth = model.backbone.n_layers
            if name.startswith('backbone.transformer.'):
                layer = int(name.split('.')[2])
                layer_factor = layer_decay ** (depth - 1 - layer)
            elif name.startswith('backbone.embedding.'):
                layer_factor = layer_decay ** depth
        rate = lr * (backbone_lr_ratio * layer_factor if role == 'backbone' else 1.)
        decay = 0. if parameter.ndim < 2 or name.endswith('.bias') else weight_decay
        key = (rate, decay, role)
        groups.setdefault(key, []).append(parameter)
    parameter_groups = [dict(params=parameters, lr=key[0], weight_decay=key[1], role=key[2])
                        for key, parameters in groups.items()]
    optimizer = torch.optim.AdamW(parameter_groups)
    description = [dict(lr=g['lr'], weight_decay=g['weight_decay'], role=g['role'],
                        parameters=sum(p.numel() for p in g['params'])) for g in parameter_groups]
    return optimizer, description


def make_scheduler(optimizer, steps, schedule='cosine', warmup_fraction=.1):
    if steps < 1 or schedule not in ('constant', 'cosine') or not 0 <= warmup_fraction < 1:
        raise ValueError('invalid scheduler configuration')
    warmup_steps = int(steps * warmup_fraction)
    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == 'constant':
            return 1.
        progress = min(1., max(0., (step - warmup_steps) / max(steps - warmup_steps, 1)))
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
