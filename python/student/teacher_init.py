"""Load Qwen gate/up vectors for exact-arch teacher weight init (Torch R&D)."""

from __future__ import annotations

import numpy as np

from env_config import apply_huggingface_paths, resolve_teacher_device


def load_exact_teacher_vectors(
    model_name: str,
    layer: int,
    neuron: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FP32 gate/up rows for one MLP neuron (shape [hidden_size] each)."""
    apply_huggingface_paths()
    from extract_chain import load_qwen_weights

    gate_w, up_w, hidden, _, _ = load_qwen_weights(
        model_name,
        layer,
        neuron,
        resolve_teacher_device(),
    )
    if gate_w.shape != (hidden,) or up_w.shape != (hidden,):
        raise ValueError(
            f"teacher gate/up shape mismatch: gate={gate_w.shape} up={up_w.shape} "
            f"expected ({hidden},)"
        )
    return gate_w, up_w


def load_layer_teacher_weights(
    model_name: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return FP32 gate/up/down for one transformer layer FFN."""
    apply_huggingface_paths()
    from extract_layer_ffn import load_qwen_ffn_weights

    gate_w, up_w, down_w, hidden, intermediate, _ = load_qwen_ffn_weights(
        model_name,
        layer,
        resolve_teacher_device(),
    )
    if gate_w.shape != (intermediate, hidden):
        raise ValueError(f"teacher gate shape mismatch: {gate_w.shape}")
    if up_w.shape != (intermediate, hidden):
        raise ValueError(f"teacher up shape mismatch: {up_w.shape}")
    if down_w.shape != (hidden, intermediate):
        raise ValueError(f"teacher down shape mismatch: {down_w.shape}")
    return gate_w, up_w, down_w, hidden, intermediate


def svd_lowrank_factors(w: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor W [out, in] as A [out, r] @ B [r, in] via truncated SVD."""
    out_dim, in_dim = w.shape
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if rank > min(out_dim, in_dim):
        raise ValueError(
            f"rank {rank} exceeds min(out={out_dim}, in={in_dim}) for shape {w.shape}"
        )
    u, s, vh = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    s_r = s[:rank]
    a = u[:, :rank] * np.sqrt(s_r)
    b = np.sqrt(s_r)[:, np.newaxis] * vh[:rank, :]
    return a.astype(np.float32), b.astype(np.float32)
