"""
Compatibility patches that must run before `import fairseq`.

1. PyTorch 2.6+ defaults torch.load(..., weights_only=True).
   Fairseq HuBERT checkpoints unpickle custom classes; they need weights_only=False.

2. Python 3.12 raises ValueError for @dataclass instances used as mutable field
   defaults. fairseq 0.12.2, hydra, and omegaconf all use this pattern.
   We patch dataclasses._get_field to silently allow it.

3. After fix #2, hydra_init() in fairseq/__init__.py still fails because
   omegaconf cannot handle default_factory fields in structured configs.
   hydra_init() is only needed for the Hydra CLI (fairseq-train etc.) —
   not for checkpoint loading. We patch fairseq/__init__.py to skip it on error.
"""
from __future__ import annotations

import importlib.util
import inspect

_applied = False


def _patch_py312_mutable_dataclass_defaults() -> None:
    """Patch dataclasses._get_field to accept @dataclass instances as defaults.

    Python 3.12 raises ValueError when a field's default is itself a @dataclass
    instance. fairseq/hydra/omegaconf use this extensively. We intercept
    _get_field and convert such defaults to field(default_factory=...) so
    the @dataclass decorator can finish processing without error.
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
            setattr(cls, a_name, dataclasses.field(default_factory=type(raw)))
        return _orig(cls, a_name, a_type, default_kw_only)

    dataclasses._get_field = _lenient_get_field
    dataclasses._py312_compat_patched = True
    print("[fairseq_compat] patched dataclasses._get_field for Python 3.12")


def _patch_fairseq_hydra_init() -> None:
    """Wrap hydra_init() in fairseq/__init__.py with try-except.

    After the _get_field patch converts mutable defaults to default_factory,
    omegaconf sees MISSING instead of actual default instances and raises
    ValidationError inside hydra_init(). hydra_init() is only needed for
    the Hydra CLI interface — not for checkpoint loading / HuBERT inference.
    Skipping it on error allows `import fairseq` to complete normally.
    """
    spec = importlib.util.find_spec("fairseq")
    if spec is None or spec.origin is None:
        return

    from pathlib import Path

    fairseq_init = Path(spec.origin)
    text = fairseq_init.read_text("utf-8")

    if "hydra_init()" not in text:
        return  # already removed or different version

    # Check if already patched (our sentinel comment is present)
    if "# compat: skip on py312" in text:
        return

    patched = text.replace(
        "hydra_init()",
        "try:\n    hydra_init()\nexcept Exception:\n    pass  # compat: skip on py312",
    )

    if patched == text:
        return

    fairseq_init.write_text(patched, "utf-8")
    for pyc in fairseq_init.parent.glob("__pycache__/__init__*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass
    print("[fairseq_compat] patched fairseq/__init__.py: hydra_init() wrapped in try-except")


def apply_fairseq_torch_load_compat() -> None:
    global _applied
    if _applied:
        return

    _patch_py312_mutable_dataclass_defaults()
    _patch_fairseq_hydra_init()

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
