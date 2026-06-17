"""Shared-bottleneck v0 micro-net: one gate/up block, per-chain head and alpha."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rng import init_model_weights
from .ternary import ternary_linear


def shared_v0_param_count(input_dim: int, hidden_dim: int, chain_group: int) -> int:
    if chain_group < 1:
        raise ValueError(f"chain_group must be >= 1, got {chain_group}")
    return 2 * hidden_dim * input_dim + chain_group * hidden_dim + chain_group


class SharedMicroNetV0(nn.Module):
    """v0 SwiGLU bottleneck shared across K scalar chains; per-chain head + alpha."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        chain_group: int,
        *,
        use_ternary: bool = True,
        ternary_threshold: float = 0.0,
        ste_mode: str = "plain",
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if chain_group < 1:
            raise ValueError(f"chain_group must be >= 1, got {chain_group}")
        if ste_mode not in {"plain", "masked"}:
            raise ValueError(f"unsupported ste_mode: {ste_mode}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.chain_group = chain_group
        self.use_ternary = use_ternary
        self.ternary_threshold = ternary_threshold
        self.ste_mode = ste_mode

        self.gate = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.up = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.head = nn.Parameter(torch.empty(chain_group, hidden_dim))
        self.alpha = nn.Parameter(torch.ones(chain_group, dtype=torch.float32))
        self._init_lcg_weights(init_scale)

    def _init_lcg_weights(self, scale: float) -> None:
        gate_size = self.hidden_dim * self.input_dim
        head_size = self.chain_group * self.hidden_dim
        w = init_model_weights(gate_size, head_size, 0, scale)
        with torch.no_grad():
            self.gate.copy_(torch.from_numpy(w["gate"]).reshape(self.hidden_dim, self.input_dim))
            self.up.copy_(torch.from_numpy(w["up"]).reshape(self.hidden_dim, self.input_dim))
            self.head.copy_(torch.from_numpy(w["head"]).reshape(self.chain_group, self.hidden_dim))
            self.alpha.fill_(1.0)

    def param_count_total(self) -> int:
        return shared_v0_param_count(self.input_dim, self.hidden_dim, self.chain_group)

    def _linear(self, x: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
        return ternary_linear(
            x,
            shadow,
            use_ternary=self.use_ternary,
            training=self.training,
            threshold=self.ternary_threshold,
            ste_mode=self.ste_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim] -> [B, chain_group] predictions."""
        gate = self._linear(x, self.gate)
        up = self._linear(x, self.up)
        h0 = F.silu(gate) * up
        y_tern = self._linear(h0, self.head)
        return y_tern * self.alpha.unsqueeze(0)

    @torch.no_grad()
    def eval_mse(self, x: torch.Tensor, y: torch.Tensor) -> float:
        was_training = self.training
        self.eval()
        pred = self.forward(x)
        mse = F.mse_loss(pred, y, reduction="mean")
        if was_training:
            self.train()
        return float(mse.item())

    def train_batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x)
        return F.mse_loss(pred, y, reduction="mean")
