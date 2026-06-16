"""MicroNet student models matching BatchMicroNet v0/v1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rng import init_model_weights
from .ternary import ternary_linear


def param_count(arch: str, input_dim: int, hidden_dim: int) -> int:
    block1 = 2 * hidden_dim * input_dim
    block2 = 2 * hidden_dim * hidden_dim if arch == "v1" else 0
    return block1 + block2 + hidden_dim + 1


class MicroNet(nn.Module):
    """Ternary SwiGLU micro-network (v0 or v1)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        arch: str = "v0",
        use_ternary: bool = True,
        ternary_threshold: float = 0.0,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if arch not in {"v0", "v1"}:
            raise ValueError(f"unsupported arch: {arch}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.arch = arch
        self.use_ternary = use_ternary
        self.ternary_threshold = ternary_threshold

        gate_size = hidden_dim * input_dim
        block2_size = hidden_dim * hidden_dim if arch == "v1" else 0

        self.gate = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.up = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.head = nn.Parameter(torch.empty(1, hidden_dim))
        self.gate2 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim) if arch == "v1" else torch.empty(0))
        self.up2 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim) if arch == "v1" else torch.empty(0))
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self._init_lcg_weights(gate_size, hidden_dim, block2_size, init_scale)

    def _init_lcg_weights(
        self,
        gate_size: int,
        head_size: int,
        block2_size: int,
        scale: float,
    ) -> None:
        w = init_model_weights(gate_size, head_size, block2_size, scale)
        with torch.no_grad():
            self.gate.copy_(torch.from_numpy(w["gate"]).reshape(self.hidden_dim, self.input_dim))
            self.up.copy_(torch.from_numpy(w["up"]).reshape(self.hidden_dim, self.input_dim))
            self.head.copy_(torch.from_numpy(w["head"]).reshape(1, self.hidden_dim))
            if self.arch == "v1":
                self.gate2.zero_()
                self.up2.zero_()

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
        gate = self._linear(x, self.gate)
        up = self._linear(x, self.up)
        h0 = F.silu(gate) * up

        if self.arch == "v1":
            gate2 = self._linear(h0, self.gate2)
            up2 = self._linear(h0, self.up2)
            h1 = h0 + F.silu(gate2) * up2
            y_tern = self._linear(h1, self.head).squeeze(-1)
        else:
            y_tern = self._linear(h0, self.head).squeeze(-1)

        return self.alpha * y_tern

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
