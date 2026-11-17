"""Model definitions for shrimp storage-time prediction from 2T2D spectra.

The module contains the proposed LTC-Attn model and the comparison
architectures reported in the manuscript:

============ ==========================================================
Name         Architecture
============ ==========================================================
LTC-Attn     liquid time-constant layer + multi-head attention pooling
LSTM-Attn    LSTM + multi-head attention pooling
GRU-Attn     GRU + multi-head attention pooling
ResNet       ResNet-18 on 2D correlation spectra (306 x 306, 2 channels)
MobileNet    MobileNetV2 on 2D correlation spectra
ViT          Vision Transformer on 2D correlation spectra
============ ==========================================================

Input convention
----------------
Image-based models (ResNet, MobileNet, ViT) receive the two-dimensional
correlation spectra as ``[batch, 2, 306, 306]``, the two channels being the
synchronous and the asynchronous spectrum.

Sequence-based models (LTC-Attn, LSTM-Attn, GRU-Attn) receive the same tensor
and internally reshape it to ``[batch, 306, 612]``, i.e. one row of the
correlation matrix per step. The row index runs along the vertical band
coordinate, so the sequence order follows the Raman shift axis.

The 1D ablation in the manuscript uses the *same* LTC-Attn class with
``input_dim=1`` and an input of shape ``[batch, 1, 306, 1]``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torchvision.models import mobilenet_v2, resnet18

#: Number of Raman bands in the retained range (600-1800 cm^-1).
N_BANDS = 306

#: Channels of the input tensor: synchronous and asynchronous spectrum.
N_CHANNELS = 2


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class LiquidNeuronLayer(nn.Module):
    """Liquid time-constant (LTC) recurrent layer.

    The layer implements the liquid time-constant formulation of Hasani et al.,
    "Liquid Time-constant Networks", AAAI 2021. The hidden state of every
    neuron evolves according to Eq. (5) of the manuscript:

        dx_t/dt = -(1/tau + f(x_t, I_t, t, theta)) * x_t
                  +  f(x_t, I_t, t, theta) * A

        f(x_t, I_t, t, theta) = sigmoid(W_a x_t + W_b I_t + b)

    with

        x_t     hidden state of a neuron at band t       (code: ``h``)
        I_t     feature input at band t                  (code: ``x_input``)
        t       pseudo-time, i.e. the band index         (integration variable)
        tau     time constant, positive and learnable    (code: ``self.tau``)
        A       reversal potential, learnable vector     (code: ``self.A``)
        theta   learnable weights of the gate network    (code: ``recurrent_fc``,
                ``input_fc``, ``bias``)

    A small ``tau`` makes a neuron respond quickly, while a large gate value
    pulls its state towards ``A``. The equation is integrated with a
    fourth-order Runge-Kutta step over one unit interval per band, so the state
    is advanced once for every Raman shift position. ``t`` therefore enters
    only as that integration variable and is not an explicit input of the gate.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_fc = nn.Linear(hidden_size, hidden_size)
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size))
        self.A = nn.Parameter(torch.ones(hidden_size))
        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))

    def dynamics(self, h: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        """Right-hand side of the ODE for one band."""
        gate = torch.sigmoid(self.recurrent_fc(h) + self.input_fc(x_input) + self.bias)
        tau = F.softplus(self.tau) + 1e-3
        return -(1.0 / tau + gate) * h + gate * self.A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate the ODE along the sequence axis.

        Parameters
        ----------
        x
            Sequence of shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        torch.Tensor
            Hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        batch, seq_len, _ = x.shape
        h = self.h0.expand(batch, self.hidden_size)
        integration_times = torch.tensor([0.0, 1.0], device=x.device)

        states = []
        for step in range(seq_len):
            x_t = x[:, step, :]

            def ode_func(_time, state, x_t=x_t):
                return self.dynamics(state, x_t)

            h = odeint(ode_func, h, integration_times, method="rk4")[-1]
            states.append(h)

        return torch.stack(states, dim=1)


class MultiHeadAttnPool(nn.Module):
    """Multi-head attention pooling over the sequence axis.

    A single query, obtained from the mean hidden state, attends over all
    steps, so the resulting weights can be read as per-band importance.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = self.head_dim ** -0.5

    def forward(
        self, h_seq: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        batch, steps, hidden = h_seq.shape

        query = self.query(h_seq.mean(dim=1, keepdim=True))
        key = self.key(h_seq)
        value = self.value(h_seq)

        query = query.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)

        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).contiguous().view(batch, 1, hidden)
        out = self.out_proj(context.squeeze(1))

        if return_attn:
            mean_weights = weights.mean(dim=1).squeeze(1)
            mean_weights = mean_weights - mean_weights.min(dim=1, keepdim=True)[0]
            mean_weights = mean_weights / (
                mean_weights.max(dim=1, keepdim=True)[0] + 1e-8
            )
            return out, mean_weights
        return out


def create_regressor(hidden_size: int, dropout_rate: float = 0.1) -> nn.Sequential:
    """Three-layer fully connected regression head."""
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Dropout(dropout_rate / 2),
        nn.Linear(64, 1),
    )


class _Squeeze1(nn.Module):
    """Drop the trailing singleton dimension so every model returns ``(batch,)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1)


def _to_sequence(x: torch.Tensor) -> torch.Tensor:
    """Reshape ``(batch, channels, bands, bands)`` into ``(batch, bands, 612)``."""
    batch, channels, rows, cols = x.shape
    return x.permute(0, 2, 1, 3).reshape(batch, rows, channels * cols)


# --------------------------------------------------------------------------- #
# Sequence-based models
# --------------------------------------------------------------------------- #

class LNN_2T2D_Model(nn.Module):
    """Proposed LTC-Attn model: LTC layer + attention pooling + regressor."""

    def __init__(
        self,
        input_dim: int = N_CHANNELS * N_BANDS,
        hidden_size: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.liquid_layer = LiquidNeuronLayer(input_dim, hidden_size)
        self.pool = MultiHeadAttnPool(hidden_size, num_heads=4)
        self.regressor = create_regressor(hidden_size, dropout_rate)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        h_seq = self.liquid_layer(_to_sequence(x))
        if return_attn:
            features, weights = self.pool(h_seq, return_attn=True)
            return self.regressor(features).squeeze(1), weights
        return self.regressor(self.pool(h_seq)).squeeze(1)


class LSTM_Attn_Model(nn.Module):
    """LSTM + attention pooling."""

    def __init__(
        self,
        input_dim: int = N_CHANNELS * N_BANDS,
        hidden_size: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.pool = MultiHeadAttnPool(hidden_size, num_heads=4)
        self.regressor = create_regressor(hidden_size, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.lstm(_to_sequence(x))
        return self.regressor(self.pool(h_seq)).squeeze(1)


class GRU_Attn_Model(nn.Module):
    """GRU + attention pooling."""

    def __init__(
        self,
        input_dim: int = N_CHANNELS * N_BANDS,
        hidden_size: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.pool = MultiHeadAttnPool(hidden_size, num_heads=4)
        self.regressor = create_regressor(hidden_size, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.gru(_to_sequence(x))
        return self.regressor(self.pool(h_seq)).squeeze(1)


# --------------------------------------------------------------------------- #
# Image-based models
# --------------------------------------------------------------------------- #

class ViTRegression(nn.Module):
    """Vision Transformer with a single regression output."""

    def __init__(
        self,
        img_size: int = N_BANDS,
        patch_size: int = 6,
        in_chans: int = N_CHANNELS,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.fc(self.norm(x[:, 0])).squeeze(1)


def build_resnet(dropout_rate: float = 0.1) -> nn.Module:
    """ResNet-18 accepting two-channel correlation spectra."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(N_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate), nn.Linear(512, 1), _Squeeze1()
    )
    return model


def build_mobilenet(dropout_rate: float = 0.1) -> nn.Module:
    """MobileNetV2 accepting two-channel correlation spectra."""
    model = mobilenet_v2(weights=None)
    model.features[0][0] = nn.Conv2d(N_CHANNELS, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.last_channel, 1),
        _Squeeze1(),
    )
    return model


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def get_model(name: str, dropout_rate: float = 0.1, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    ``LTC-Attn`` accepts ``input_dim=1`` for the one-dimensional ablation; the
    input then has shape ``(batch, 1, 306, 1)`` and the sequence length stays
    306 with one feature per step.
    """
    key = name.lower().replace("-", "_")

    if key in {"ltc_attn", "lnn_2t2d"}:
        return LNN_2T2D_Model(dropout_rate=dropout_rate, **kwargs)
    if key in {"lstm_attn"}:
        return LSTM_Attn_Model(dropout_rate=dropout_rate, **kwargs)
    if key in {"gru_attn"}:
        return GRU_Attn_Model(dropout_rate=dropout_rate, **kwargs)
    if key in {"vit", "vit_regression"}:
        return ViTRegression(dropout_rate=dropout_rate, **kwargs)
    if key in {"resnet", "resnet18"}:
        return build_resnet(dropout_rate=dropout_rate)
    if key in {"mobilenet", "mobilenetv2"}:
        return build_mobilenet(dropout_rate=dropout_rate)

    raise ValueError(f"unknown model name: {name!r}")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # batch size 1 keeps the ViT self-attention matrix within CPU memory
    batch = torch.randn(1, N_CHANNELS, N_BANDS, N_BANDS)
    for model_name in ("LTC-Attn", "LSTM-Attn", "GRU-Attn", "ViT", "ResNet", "MobileNet"):
        net = get_model(model_name)
        print(f"{model_name:<11} params={count_parameters(net):>9,} out={net(batch).shape}")
