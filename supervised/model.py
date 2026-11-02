"""Label-trained heads on the original Kronos causal Transformer."""
from __future__ import annotations

import torch
from torch import nn

from model import Kronos, KronosTokenizer


def kronos_config(model):
    names = ('s1_bits', 's2_bits', 'n_layers', 'd_model', 'n_heads', 'ff_dim',
             'ffn_dropout_p', 'attn_dropout_p', 'resid_dropout_p',
             'token_dropout_p', 'learn_te')
    return {name: getattr(model, name) for name in names}


def tiny_config():
    return dict(s1_bits=4, s2_bits=4, n_layers=2, d_model=64, n_heads=4,
                ff_dim=128, ffn_dropout_p=0., attn_dropout_p=0.,
                resid_dropout_p=0., token_dropout_p=0., learn_te=True)


def causal_mean(h, width):
    """Trailing mean including the current observation, with no future/pad reads."""
    if width < 1:
        raise ValueError('temporal scale must be positive')
    prefix = torch.cat([torch.zeros_like(h[:, :1]), h.cumsum(dim=1)], dim=1)
    ends = torch.arange(1, h.shape[1] + 1, device=h.device)
    starts = (ends - width).clamp_min(0)
    return (prefix[:, ends] - prefix[:, starts]) / (ends - starts)[None, :, None]


class SupervisedKronos(nn.Module):
    """Point logits [B,T] or sequence logits [B], never next-token logits.

    Right padding is safe because all upstream self attention is causal. For
    sequence classification uses the last valid state and optional masked pooling. Tokenizer is
    frozen; discrete quantization is not presented as differentiable fine tuning.
    """

    def __init__(self, backbone, n_features, task='point', input_kind='features',
                 tokenizer=None, freeze_backbone=False, head_type='mlp',
                 sequence_pooling='last', scales=(4, 16, 64), head_dropout=0.):
        super().__init__()
        if task not in ('point', 'sequence'):
            raise ValueError('task must be point or sequence')
        if input_kind not in ('features', 'ohlcv'):
            raise ValueError('input_kind must be features or ohlcv')
        if head_type not in ('mlp', 'multiscale') or sequence_pooling not in ('last', 'attention'):
            raise ValueError('invalid head type or sequence pooling')
        if not scales or any(not isinstance(s, int) or s < 1 for s in scales):
            raise ValueError('scales must contain positive integers')
        if not 0 <= head_dropout < 1:
            raise ValueError('head dropout must lie in [0,1)')
        if input_kind == 'ohlcv':
            if n_features != 6 or tokenizer is None or tokenizer.d_in != 6:
                raise ValueError('ohlcv requires six genuine OHLCVA features and a tokenizer')
            if (backbone.s1_bits, backbone.s2_bits) != (tokenizer.s1_bits, tokenizer.s2_bits):
                raise ValueError('tokenizer and backbone codebooks do not match')
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.task, self.input_kind = task, input_kind
        self.n_features = n_features
        self.freeze_backbone = freeze_backbone
        self.head_type, self.sequence_pooling = head_type, sequence_pooling
        self.scales, self.head_dropout = tuple(scales), head_dropout
        self.adapter = (nn.Sequential(nn.Linear(n_features, backbone.d_model), nn.GELU(),
                                      nn.LayerNorm(backbone.d_model))
                        if input_kind == 'features' else None)
        self.pool_score = (nn.Linear(backbone.d_model, 1, bias=False)
                           if task == 'sequence' and sequence_pooling == 'attention' else None)
        head_dim = backbone.d_model
        if task == 'point' and head_type == 'multiscale':
            head_dim *= 1 + len(scales)
        if self.pool_score is not None:
            head_dim *= 2  # last valid state + masked attention summary
        self.head_drop = nn.Dropout(head_dropout)
        self.classifier = nn.Sequential(nn.Linear(head_dim, backbone.d_model // 2),
                                        nn.GELU(), nn.Linear(backbone.d_model // 2, 1))
        if freeze_backbone:
            backbone.requires_grad_(False)
        # Forecast token heads and calendar embeddings are not used by this task.
        for part in (backbone.head, backbone.dep_layer, backbone.time_emb):
            part.requires_grad_(False)
        if input_kind == 'features':
            backbone.embedding.requires_grad_(False)
        if tokenizer is not None:
            tokenizer.requires_grad_(False)
            tokenizer.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.tokenizer is not None:
            self.tokenizer.eval()
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, lengths):
        if x.ndim != 3 or x.shape[-1] != self.n_features:
            raise ValueError('expected X[B,T,n_features]')
        if torch.any(lengths < 1) or torch.any(lengths > x.shape[1]):
            raise ValueError('invalid sequence lengths')
        if self.input_kind == 'ohlcv':
            with torch.no_grad():
                ids = self.tokenizer.encode(x, half=True)
            h = self.backbone.embedding(ids)
        else:
            h = self.adapter(x)
        h = self.backbone.token_drop(h)
        for layer in self.backbone.transformer:
            h = layer(h)
        h = self.backbone.norm(h)
        if self.task == 'sequence':
            last = h[torch.arange(len(h), device=h.device), lengths - 1]
            if self.pool_score is not None:
                valid = torch.arange(h.shape[1], device=h.device)[None] < lengths[:, None]
                weights = self.pool_score(h).squeeze(-1).masked_fill(~valid, float('-inf')).softmax(-1)
                h = torch.cat([last, (weights.unsqueeze(-1) * h).sum(dim=1)], dim=-1)
            else:
                h = last
        elif self.head_type == 'multiscale':
            h = torch.cat([h] + [h - causal_mean(h, s) for s in self.scales], dim=-1)
        return self.classifier(self.head_drop(h)).squeeze(-1)

    def specification(self):
        return dict(backbone=kronos_config(self.backbone), n_features=self.n_features,
                    task=self.task, input_kind=self.input_kind,
                    freeze_backbone=self.freeze_backbone,
                    head_type=self.head_type, sequence_pooling=self.sequence_pooling,
                    scales=list(self.scales), head_dropout=self.head_dropout,
                    tokenizer=dict(self.tokenizer._hub_mixin_config) if self.tokenizer is not None else None)


def from_specification(spec):
    spec = dict(spec)
    backbone = Kronos(**spec.pop('backbone'))
    config = spec.pop('tokenizer')
    tokenizer = KronosTokenizer(**config) if config is not None else None
    return SupervisedKronos(backbone, tokenizer=tokenizer, **spec)
