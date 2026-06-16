"""MicroNet student models: v0/v1/v1b (bottleneck H) or exact (teacher-shaped scalar SwiGLU)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rng import init_model_weights
from .ternary import ternary_linear

_V1_ARCHES = frozenset({"v1", "v1b"})
_ALL_ARCHES = frozenset({"v0", "v1", "v1b", "exact"})


def param_count(arch: str, input_dim: int, hidden_dim: int) -> int:
    if arch == "exact":
        return 2 * input_dim + 1
    block1 = 2 * hidden_dim * input_dim
    block2 = 2 * hidden_dim * hidden_dim if arch in _V1_ARCHES else 0
    skip = input_dim + 1 if arch == "v1b" else 0
    return block1 + block2 + hidden_dim + 1 + skip


class MicroNet(nn.Module):
    """Ternary SwiGLU micro-network (v0, v1, v1b, or exact)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        arch: str = "v0",
        use_ternary: bool = True,
        ternary_threshold: float = 0.0,
        init_scale: float = 0.1,
        block2_init: str = "zero",
        block2_init_scale: float | None = None,
    ) -> None:
        super().__init__()
        if arch not in _ALL_ARCHES:
            raise ValueError(f"unsupported arch: {arch}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.arch = arch
        self.use_ternary = use_ternary
        self.ternary_threshold = ternary_threshold
        if block2_init not in {"zero", "lcg"}:
            raise ValueError(f"unsupported block2_init: {block2_init}")
        self.block2_init = block2_init
        self.block2_init_scale = block2_init_scale

        if arch == "exact":
            self.gate = nn.Parameter(torch.empty(1, input_dim))
            self.up = nn.Parameter(torch.empty(1, input_dim))
            self.head = nn.Parameter(torch.empty(0))
            self.gate2 = nn.Parameter(torch.empty(0))
            self.up2 = nn.Parameter(torch.empty(0))
            self.w_res = nn.Parameter(torch.empty(0))
            self.beta = nn.Parameter(torch.empty(0))
            self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self._init_exact_weights(init_scale)
            return

        gate_size = hidden_dim * input_dim
        block2_size = hidden_dim * hidden_dim if arch in _V1_ARCHES else 0
        w_res_size = input_dim if arch == "v1b" else 0

        self.gate = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.up = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.head = nn.Parameter(torch.empty(1, hidden_dim))
        self.gate2 = nn.Parameter(
            torch.zeros(hidden_dim, hidden_dim) if arch in _V1_ARCHES else torch.empty(0)
        )
        self.up2 = nn.Parameter(
            torch.zeros(hidden_dim, hidden_dim) if arch in _V1_ARCHES else torch.empty(0)
        )
        self.w_res = nn.Parameter(torch.zeros(1, input_dim) if arch == "v1b" else torch.empty(0))
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32) if arch == "v1b" else torch.empty(0))

        self._init_lcg_weights(gate_size, hidden_dim, block2_size, init_scale, w_res_size)

    def _init_exact_weights(self, scale: float) -> None:
        """LCG init for gate/up vectors (D each); matches teacher param layout."""
        w = init_model_weights(self.input_dim, 0, 0, scale)
        with torch.no_grad():
            self.gate.copy_(torch.from_numpy(w["gate"]).reshape(1, self.input_dim))
            self.up.copy_(torch.from_numpy(w["up"]).reshape(1, self.input_dim))

    def _init_lcg_weights(
        self,
        gate_size: int,
        head_size: int,
        block2_size: int,
        scale: float,
        w_res_size: int,
    ) -> None:
        w = init_model_weights(
            gate_size,
            head_size,
            block2_size,
            scale,
            w_res_size=w_res_size,
            block2_init=self.block2_init if self.arch in _V1_ARCHES else "zero",
            block2_init_scale=self.block2_init_scale,
        )
        with torch.no_grad():
            self.gate.copy_(torch.from_numpy(w["gate"]).reshape(self.hidden_dim, self.input_dim))
            self.up.copy_(torch.from_numpy(w["up"]).reshape(self.hidden_dim, self.input_dim))
            self.head.copy_(torch.from_numpy(w["head"]).reshape(1, self.hidden_dim))
            if self.arch in _V1_ARCHES:
                self.gate2.copy_(torch.from_numpy(w["gate2"]).reshape(self.hidden_dim, self.hidden_dim))
                self.up2.copy_(torch.from_numpy(w["up2"]).reshape(self.hidden_dim, self.hidden_dim))
            if self.arch == "v1b":
                self.w_res.zero_()
                self.beta.fill_(1.0)

    def param_count_total(self) -> int:
        return param_count(self.arch, self.input_dim, self.hidden_dim)

    def _linear(
        self,
        x: torch.Tensor,
        shadow: torch.Tensor,
    ) -> torch.Tensor:
        return ternary_linear(
            x,
            shadow,
            use_ternary=self.use_ternary,
            training=self.training,
            threshold=self.ternary_threshold,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim] -> [B] predictions."""
        if self.arch == "exact":
            gate = self._linear(x, self.gate).squeeze(-1)
            up = self._linear(x, self.up).squeeze(-1)
            return self.alpha * F.silu(gate) * up

        gate = self._linear(x, self.gate)
        up = self._linear(x, self.up)
        h0 = F.silu(gate) * up

        if self.arch in _V1_ARCHES:
            gate2 = self._linear(h0, self.gate2)
            up2 = self._linear(h0, self.up2)
            h1 = h0 + F.silu(gate2) * up2
            y_tern = self._linear(h1, self.head).squeeze(-1)
        else:
            y_tern = self._linear(h0, self.head).squeeze(-1)

        out = self.alpha * y_tern
        if self.arch == "v1b":
            # FP32 dense skip: direct linear path from x (not ternary-quantized).
            y_skip = F.linear(x, self.w_res).squeeze(-1)
            out = out + self.beta * y_skip
        return out

    @torch.no_grad()
    def eval_mse(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Inference MSE with ternary forward when use_ternary (matches Mojo eval_mse)."""
        was_training = self.training
        self.eval()
        pred = self.forward(x)
        mse = F.mse_loss(pred, y, reduction="mean")
        if was_training:
            self.train()
        return float(mse.item())

    def train_batch_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """One batch train forward + MSE loss (mean over batch)."""
        pred = self.forward(x)
        return F.mse_loss(pred, y, reduction="mean")
