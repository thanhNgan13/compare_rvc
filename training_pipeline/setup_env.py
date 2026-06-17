"""Chỉ dùng bên trong thư mục rvc_standalone — gốc = thư mục cha của package training_pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def get_standalone_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bootstrap(clear_argv_for_config: bool = True):
    root = get_standalone_root()
    os.chdir(root)
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    for d in tuple(
        root / x
        for x in (
            "assets/weights",
            "assets/indices",
            "assets/hubert",
            "assets/pretrained",
            "assets/pretrained_v2",
            "assets/uvr5_weights",
            "assets/rmvpe",
            "logs",
            "datasets",
            "TEMP",
        )
    ):
        d.mkdir(parents=True, exist_ok=True)

    saved = sys.argv[:]
    if clear_argv_for_config:
        sys.argv = [saved[0] if saved else "python"]
    try:
        from configs.config import Config

        return root, Config()
    finally:
        sys.argv = saved
