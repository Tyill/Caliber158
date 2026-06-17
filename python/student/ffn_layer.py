"""Shared ternary FFN layer (v2 production target)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rng import init_ffn_lowrank_shadow, init_ffn_shadow
from .teacher_init import svd_lowrank_factors
from .ternary import ternary_linear


def ffn_param_count(
    input_dim: int,
    intermediate_dim: int,
    rank: int,
    *,
    output_scale: str = "none",
) -> int:
    """Total trainable params for FFN core + optional FP32 scales."""
    if rank <= 0:
        core = 3 * intermediate_dim * input_dim
    else:
        core = 3 * rank * (input_dim + intermediate_dim)
    if output_scale == "global":
        return core + 1
    if output_scale == "channel":
        return core + input_dim
    if output_scale == "projection":
        if rank <= 0:
            return core + 2 * intermediate_dim + input_dim
        return core + 2 * (rank + intermediate_dim) + (rank + input_dim)
    if output_scale != "none":
        raise ValueError(f"unsupported output_scale: {output_scale}")
    return core


class TernaryFFNLayer(nn.Module):
    """Ternary SwiGLU FFN: gate/up/down with STE (full-rank or low-rank)."""

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        *,
        rank: int = 0,
        use_ternary: bool = True,
        ternary_threshold: float = 0.0,
        ste_mode: str = "plain",
        init_scale: float = 0.1,
        weight_init: str = "lcg",
        teacher_gate: np.ndarray | None = None,
        teacher_up: np.ndarray | None = None,
        teacher_down: np.ndarray | None = None,
        output_scale: str = "none",
    ) -> None:
        super().__init__()
        if ste_mode not in {"plain", "masked"}:
            raise ValueError(f"unsupported ste_mode: {ste_mode}")
        if weight_init not in {"lcg", "teacher"}:
            raise ValueError(f"unsupported weight_init: {weight_init}")
        if output_scale not in {"none", "global", "channel", "projection"}:
            raise ValueError(f"unsupported output_scale: {output_scale}")
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}")
        if rank > 0 and rank > min(input_dim, intermediate_dim):
            raise ValueError(
                f"rank {rank} exceeds min(input_dim={input_dim}, "
                f"intermediate_dim={intermediate_dim})"
            )

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.rank = rank
        self.use_ternary = use_ternary
        self.ternary_threshold = ternary_threshold
        self.ste_mode = ste_mode
        self.weight_init = weight_init
        self.output_scale = output_scale

        if output_scale == "global":
            self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.channel_scale = nn.Parameter(torch.empty(0))
        elif output_scale == "channel":
            self.alpha = nn.Parameter(torch.empty(0))
            self.channel_scale = nn.Parameter(torch.ones(input_dim, dtype=torch.float32))
        else:
            self.alpha = nn.Parameter(torch.empty(0))
            self.channel_scale = nn.Parameter(torch.empty(0))

        if rank <= 0:
            self.gate = nn.Parameter(torch.empty(intermediate_dim, input_dim))
            self.up = nn.Parameter(torch.empty(intermediate_dim, input_dim))
            self.down = nn.Parameter(torch.empty(input_dim, intermediate_dim))
            self.gate_a = nn.Parameter(torch.empty(0))
            self.gate_b = nn.Parameter(torch.empty(0))
            self.up_a = nn.Parameter(torch.empty(0))
            self.up_b = nn.Parameter(torch.empty(0))
            self.down_a = nn.Parameter(torch.empty(0))
            self.down_b = nn.Parameter(torch.empty(0))
            if weight_init == "teacher":
                self._init_teacher(teacher_gate, teacher_up, teacher_down)
            else:
                self._init_full(init_scale)
            if output_scale == "projection":
                self.gate_proj_scale = nn.Parameter(torch.ones(intermediate_dim))
                self.up_proj_scale = nn.Parameter(torch.ones(intermediate_dim))
                self.down_proj_scale = nn.Parameter(torch.ones(input_dim))
                self._zero_lowrank_proj_scales()
            else:
                self._zero_all_proj_scales()
        else:
            self.gate = nn.Parameter(torch.empty(0))
            self.up = nn.Parameter(torch.empty(0))
            self.down = nn.Parameter(torch.empty(0))
            self.gate_a = nn.Parameter(torch.empty(intermediate_dim, rank))
            self.gate_b = nn.Parameter(torch.empty(rank, input_dim))
            self.up_a = nn.Parameter(torch.empty(intermediate_dim, rank))
            self.up_b = nn.Parameter(torch.empty(rank, input_dim))
            self.down_a = nn.Parameter(torch.empty(input_dim, rank))
            self.down_b = nn.Parameter(torch.empty(rank, intermediate_dim))
            if weight_init == "teacher":
                self._init_teacher_lowrank(teacher_gate, teacher_up, teacher_down)
            else:
                self._init_lowrank(init_scale)
            if output_scale == "projection":
                r = rank
                d = input_dim
                i = intermediate_dim
                self.gate_b_scale = nn.Parameter(torch.ones(r))
                self.gate_a_scale = nn.Parameter(torch.ones(i))
                self.up_b_scale = nn.Parameter(torch.ones(r))
                self.up_a_scale = nn.Parameter(torch.ones(i))
                self.down_b_scale = nn.Parameter(torch.ones(r))
                self.down_a_scale = nn.Parameter(torch.ones(d))
                self.gate_proj_scale = nn.Parameter(torch.empty(0))
                self.up_proj_scale = nn.Parameter(torch.empty(0))
                self.down_proj_scale = nn.Parameter(torch.empty(0))
            else:
                self._zero_all_proj_scales()
                self._zero_lowrank_proj_scales()

    def _zero_all_proj_scales(self) -> None:
        self.gate_proj_scale = nn.Parameter(torch.empty(0))
        self.up_proj_scale = nn.Parameter(torch.empty(0))
        self.down_proj_scale = nn.Parameter(torch.empty(0))

    def _zero_lowrank_proj_scales(self) -> None:
        self.gate_b_scale = nn.Parameter(torch.empty(0))
        self.gate_a_scale = nn.Parameter(torch.empty(0))
        self.up_b_scale = nn.Parameter(torch.empty(0))
        self.up_a_scale = nn.Parameter(torch.empty(0))
        self.down_b_scale = nn.Parameter(torch.empty(0))
        self.down_a_scale = nn.Parameter(torch.empty(0))

    def _init_full(self, scale: float) -> None:
        w = init_ffn_shadow(self.input_dim, self.intermediate_dim, scale)
        with torch.no_grad():
            self.gate.copy_(torch.from_numpy(w["gate"]))
            self.up.copy_(torch.from_numpy(w["up"]))
            self.down.copy_(torch.from_numpy(w["down"]))

    def _init_teacher(
        self,
        gate_w: np.ndarray | None,
        up_w: np.ndarray | None,
        down_w: np.ndarray | None,
    ) -> None:
        if gate_w is None or up_w is None or down_w is None:
            raise ValueError("teacher init requires gate/up/down weight matrices")
        i, d = self.intermediate_dim, self.input_dim
        if gate_w.shape != (i, d) or up_w.shape != (i, d) or down_w.shape != (d, i):
            raise ValueError(
                f"teacher FFN shape mismatch: gate={gate_w.shape} up={up_w.shape} "
                f"down={down_w.shape} expected gate/up=({i},{d}) down=({d},{i})"
            )
        with torch.no_grad():
            self.gate.copy_(torch.from_numpy(gate_w))
            self.up.copy_(torch.from_numpy(up_w))
            self.down.copy_(torch.from_numpy(down_w))

    def _init_teacher_lowrank(
        self,
        gate_w: np.ndarray | None,
        up_w: np.ndarray | None,
        down_w: np.ndarray | None,
    ) -> None:
        if gate_w is None or up_w is None or down_w is None:
            raise ValueError("teacher low-rank init requires gate/up/down matrices")
        r = self.rank
        gate_a, gate_b = svd_lowrank_factors(gate_w, r)
        up_a, up_b = svd_lowrank_factors(up_w, r)
        down_a, down_b = svd_lowrank_factors(down_w, r)
        with torch.no_grad():
            self.gate_a.copy_(torch.from_numpy(gate_a))
            self.gate_b.copy_(torch.from_numpy(gate_b))
            self.up_a.copy_(torch.from_numpy(up_a))
            self.up_b.copy_(torch.from_numpy(up_b))
            self.down_a.copy_(torch.from_numpy(down_a))
            self.down_b.copy_(torch.from_numpy(down_b))

    def _init_lowrank(self, scale: float) -> None:
        w = init_ffn_lowrank_shadow(
            self.input_dim, self.intermediate_dim, self.rank, scale
        )
        with torch.no_grad():
            self.gate_a.copy_(torch.from_numpy(w["gate_a"]))
            self.gate_b.copy_(torch.from_numpy(w["gate_b"]))
            self.up_a.copy_(torch.from_numpy(w["up_a"]))
            self.up_b.copy_(torch.from_numpy(w["up_b"]))
            self.down_a.copy_(torch.from_numpy(w["down_a"]))
            self.down_b.copy_(torch.from_numpy(w["down_b"]))

    def param_count_total(self) -> int:
        return ffn_param_count(
            self.input_dim,
            self.intermediate_dim,
            self.rank,
            output_scale=self.output_scale,
        )

    def projection_scale_params(self) -> list[nn.Parameter]:
        if self.output_scale != "projection":
            return []
        if self.rank <= 0:
            return [self.gate_proj_scale, self.up_proj_scale, self.down_proj_scale]
        return [
            self.gate_b_scale,
            self.gate_a_scale,
            self.up_b_scale,
            self.up_a_scale,
            self.down_b_scale,
            self.down_a_scale,
        ]

    def _apply_output_scale(self, y: torch.Tensor) -> torch.Tensor:
        if self.output_scale == "global":
            return self.alpha * y
        if self.output_scale == "channel":
            return y * self.channel_scale
        return y

    @torch.no_grad()
    def _linear_shadow(self, x: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
        return F.linear(x, shadow)

    @torch.no_grad()
    def _linear_ternary_eval(self, x: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
        return ternary_linear(
            x,
            shadow,
            use_ternary=True,
            training=False,
            threshold=self.ternary_threshold,
            ste_mode=self.ste_mode,
        )

    @torch.no_grad()
    def _fit_matmul_scale(
        self,
        x: torch.Tensor,
        shadow: torch.Tensor,
        scale: nn.Parameter,
    ) -> None:
        from .layer_scales import fit_projection_scales

        y_shadow = self._linear_shadow(x, shadow)
        y_tern = self._linear_ternary_eval(x, shadow)
        scale.copy_(fit_projection_scales(y_shadow, y_tern))

    @torch.no_grad()
    def fit_projection_scales_from_samples(
        self,
        x: torch.Tensor,
        *,
        max_samples: int = 4096,
    ) -> None:
        """Fit per-matmul FP32 scales: ternary output -> shadow output (BitNet-style)."""
        if self.output_scale != "projection":
            raise ValueError("fit_projection_scales_from_samples requires output_scale=projection")
        n = min(int(x.shape[0]), max_samples)
        if n <= 0:
            raise ValueError("fit_projection_scales_from_samples requires at least one row")
        x = x[:n]
        was_training = self.training
        self.eval()
        if self.rank <= 0:
            self._fit_matmul_scale(x, self.gate, self.gate_proj_scale)
            self._fit_matmul_scale(x, self.up, self.up_proj_scale)
            gate = self._linear_ternary_eval(x, self.gate) * self.gate_proj_scale
            up = self._linear_ternary_eval(x, self.up) * self.up_proj_scale
            h = F.silu(gate) * up
            y_shadow = self._linear_shadow(h, self.down)
            y_tern = self._linear_ternary_eval(h, self.down)
            from .layer_scales import fit_projection_scales

            self.down_proj_scale.copy_(fit_projection_scales(y_shadow, y_tern))
        else:
            self._fit_matmul_scale(x, self.gate_b, self.gate_b_scale)
            h1 = self._linear_ternary_eval(x, self.gate_b) * self.gate_b_scale
            self._fit_matmul_scale(h1, self.gate_a, self.gate_a_scale)
            gate = self._linear_ternary_eval(h1, self.gate_a) * self.gate_a_scale

            self._fit_matmul_scale(x, self.up_b, self.up_b_scale)
            h2 = self._linear_ternary_eval(x, self.up_b) * self.up_b_scale
            self._fit_matmul_scale(h2, self.up_a, self.up_a_scale)
            up = self._linear_ternary_eval(h2, self.up_a) * self.up_a_scale

            h = F.silu(gate) * up
            self._fit_matmul_scale(h, self.down_b, self.down_b_scale)
            h3 = self._linear_ternary_eval(h, self.down_b) * self.down_b_scale
            self._fit_matmul_scale(h3, self.down_a, self.down_a_scale)
        if was_training:
            self.train()

    @torch.no_grad()
    def fit_output_scale_from_batch(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        *,
        use_shadow: bool = True,
    ) -> None:
        """Fit FP32 output scale(s); default uses shadow (FP32) forward for stable init."""
        from .layer_scales import fit_channel_scales, fit_global_scale

        was_training = self.training
        self.eval()
        prev_ternary = self.use_ternary
        if use_shadow:
            self.use_ternary = False
        core = self._forward_core(x)
        if use_shadow:
            self.use_ternary = prev_ternary
        if self.output_scale == "global":
            self.alpha.fill_(fit_global_scale(core, target))
        elif self.output_scale == "channel":
            self.channel_scale.copy_(fit_channel_scales(core, target))
        if was_training:
            self.train()

    @torch.no_grad()
    def fit_output_scale_from_samples(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        *,
        max_samples: int = 4096,
        use_shadow: bool = True,
    ) -> None:
        """Fit output global/channel scales on up to max_samples rows."""
        n = min(int(x.shape[0]), max_samples)
        if n <= 0:
            raise ValueError("fit_output_scale_from_samples requires at least one row")
        self.fit_output_scale_from_batch(x[:n], target[:n], use_shadow=use_shadow)

    def _linear(
        self,
        x: torch.Tensor,
        shadow: torch.Tensor,
        proj_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = ternary_linear(
            x,
            shadow,
            use_ternary=self.use_ternary,
            training=self.training,
            threshold=self.ternary_threshold,
            ste_mode=self.ste_mode,
        )
        if proj_scale is not None and proj_scale.numel() > 0:
            return y * proj_scale
        return y

    def _proj(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor | None = None,
        a_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Low-rank: x [B,D] -> [B,r] -> [B,I] with optional per-matmul scales."""
        h = self._linear(x, b, b_scale)
        return self._linear(h, a, a_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_output_scale(self._forward_core(x))

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_scale == "projection":
            if self.rank <= 0:
                gate = self._linear(x, self.gate, self.gate_proj_scale)
                up = self._linear(x, self.up, self.up_proj_scale)
                h = F.silu(gate) * up
                return self._linear(h, self.down, self.down_proj_scale)
            gate = self._proj(
                x, self.gate_a, self.gate_b, self.gate_b_scale, self.gate_a_scale
            )
            up = self._proj(x, self.up_a, self.up_b, self.up_b_scale, self.up_a_scale)
            h = F.silu(gate) * up
            return self._proj(
                h, self.down_a, self.down_b, self.down_b_scale, self.down_a_scale
            )
        if self.rank <= 0:
            gate = self._linear(x, self.gate)
            up = self._linear(x, self.up)
        else:
            gate = self._proj(x, self.gate_a, self.gate_b)
            up = self._proj(x, self.up_a, self.up_b)
        h = F.silu(gate) * up
        if self.rank <= 0:
            return self._linear(h, self.down)
        return self._proj(h, self.down_a, self.down_b)

    def train_batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x)
        return F.mse_loss(pred, y, reduction="mean")

    @torch.no_grad()
    def eval_mse(self, x: torch.Tensor, y: torch.Tensor) -> float:
        was_training = self.training
        self.eval()
        pred = self.forward(x)
        mse = F.mse_loss(pred, y, reduction="mean")
        if was_training:
            self.train()
        return float(mse.item())
