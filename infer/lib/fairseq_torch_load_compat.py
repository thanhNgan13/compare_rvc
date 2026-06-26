"""
Compatibility patches that must run before `import fairseq`.

1. PyTorch 2.6+ defaults torch.load(..., weights_only=True).
   Fairseq HuBERT checkpoints unpickle custom classes; they need weights_only=False.

2. Python 3.12 raises ValueError for @dataclass instances used as mutable field defaults.
   fairseq 0.12.2 FairseqConfig has e.g. `common: CommonConfig = CommonConfig()`.
   We rewrite configs.py on disk replacing those with field(default_factory=...).
"""
from __future__ import annotations

import inspect

_applied = False


def _patch_fairseq_dataclass_configs() -> None:
    import importlib.util
    import re
    from pathlib import Path

    # Find fairseq's install location without importing it
    spec = importlib.util.find_spec("fairseq")
    if spec is None or spec.origin is None:
        return
    configs = Path(spec.origin).parent / "dataclass" / "configs.py"
    if not configs.exists():
        return

    text = configs.read_text("utf-8")

    # Check if there are any mutable dataclass defaults still present
    # Pattern:  fieldname: ClassName = ClassName()  (type annotation == default class)
    pattern = re.compile(r"(\w+): (\w+) = \2\(\)")
    if not pattern.search(text):
        return  # already patched or nothing to do

    # Ensure 'field' is in the dataclasses import line
    if re.search(r"from dataclasses import[^\n]*\bfield\b", text) is None:
        text = re.sub(
            r"(from dataclasses import )(\w)",
            r"\1field, \2",
            text,
            count=1,
        )

    # Replace:  foo: SomeClass = SomeClass()
    # With:     foo: SomeClass = field(default_factory=SomeClass)
    patched = pattern.sub(r"\1: \2 = field(default_factory=\2)", text)

    configs.write_text(patched, "utf-8")

    # Invalidate cached .pyc so Python recompiles from the patched source
    for pyc in configs.parent.glob("__pycache__/configs*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass

    print("[fairseq_compat] patched fairseq/dataclass/configs.py for Python 3.12")


def apply_fairseq_torch_load_compat() -> None:
    global _applied
    if _applied:
        return

    _patch_fairseq_dataclass_configs()

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
