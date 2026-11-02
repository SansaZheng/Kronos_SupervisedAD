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


class SupervisedKronos(nn.Module):
    """Point logits [B,T] or sequence logits [B], never next-token logits.

    Right padding is safe because all upstream self attention is causal. For
    sequence classification the final *valid* state is gathered. Tokenizer is
    frozen; discrete quantization is not presented as differentiable fine tuning.
    """

    def __init__(self, backbone, n_features, task='point', input_kind='features',
                 tokenizer=None, freeze_backbone=False):
        super().__init__()
        if task not in ('point', 'sequence'):
            raise ValueError('task must be point or sequence')
        if input_kind not in ('features', 'ohlcv'):
            raise ValueError('input_kind must be features or ohlcv')
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
        self.adapter = (nn.Sequential(nn.Linear(n_features, backbone.d_model), nn.GELU(),
                                      nn.LayerNorm(backbone.d_model))
                        if input_kind == 'features' else None)
        self.classifier = nn.Sequential(nn.Linear(backbone.d_model, backbone.d_model // 2),
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
            h = h[torch.arange(len(h), device=h.device), lengths - 1]
        return self.classifier(h).squeeze(-1)

    def specification(self):
        return dict(backbone=kronos_config(self.backbone), n_features=self.n_features,
                    task=self.task, input_kind=self.input_kind,
                    freeze_backbone=self.freeze_backbone,
                    tokenizer=dict(self.tokenizer._hub_mixin_config) if self.tokenizer is not None else None)


def from_specification(spec):
    spec = dict(spec)
    backbone = Kronos(**spec.pop('backbone'))
    config = spec.pop('tokenizer')
    tokenizer = KronosTokenizer(**config) if config is not None else None
    return SupervisedKronos(backbone, tokenizer=tokenizer, **spec)
