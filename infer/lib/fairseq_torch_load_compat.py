"""
Compatibility patches that must run before `import fairseq`.

1. PyTorch 2.6+ defaults torch.load(..., weights_only=True).
   Fairseq HuBERT checkpoints unpickle custom classes; they need weights_only=False.

2. Python 3.12 raises ValueError for @dataclass instances used as mutable field
   defaults (e.g. `common: CommonConfig = CommonConfig()`).
   fairseq 0.12.2 AND its dependency hydra both use this pattern extensively.
   Patching individual source files is fragile — instead we patch
   dataclasses._get_field itself so every affected package is fixed at once.
"""
from __future__ import annotations

import inspect

_applied = False


def _patch_py312_mutable_dataclass_defaults() -> None:
    """Patch Python 3.12 dataclasses to accept @dataclass instances as defaults.

    Converts  `foo: Cls = Cls()`  to  `foo: Cls = field(default_factory=Cls)`
    transparently at class-creation time, before Python's strict check runs.
    Covers fairseq, hydra, omegaconf, and any other package with the same pattern.
    """
    import dataclasses
    import sys
    import types

    if sys.version_info < (3, 12):
        return
    if getattr(dataclasses, "_py312_compat_patched", False):
        return

    _orig = dataclasses._get_field

    def _lenient_get_field(cls, a_name, a_type, default_kw_only):
        raw = getattr(cls, a_name, dataclasses.MISSING)
        if (
            raw is not dataclasses.MISSING
            and not isinstance(raw, (dataclasses.Field, types.MemberDescriptorType))
            and hasattr(raw, "__dataclass_fields__")
        ):
            # Mutable @dataclass default → convert to default_factory
            setattr(cls, a_name, dataclasses.field(default_factory=type(raw)))
        return _orig(cls, a_name, a_type, default_kw_only)

    dataclasses._get_field = _lenient_get_field
    dataclasses._py312_compat_patched = True
    print("[fairseq_compat] patched dataclasses._get_field for Python 3.12")


def apply_fairseq_torch_load_compat() -> None:
    global _applied
    if _applied:
        return

    _patch_py312_mutable_dataclass_defaults()

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
