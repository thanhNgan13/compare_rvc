"""
PyTorch 2.6+ defaults torch.load(..., weights_only=True).
Fairseq Hubert checkpoints unpickle custom classes; they need weights_only=False.
Apply once before any `import fairseq` that loads hubert_base.pt.
"""
from __future__ import annotations

import inspect

_applied = False


def apply_fairseq_torch_load_compat() -> None:
    global _applied
    if _applied:
        return
    import torch

    if "weights_only" not in inspect.signature(torch.load).parameters:
        _applied = True
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load
    _applied = True
