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
