"""
Hyperparameters for training (mirror WebUI "Train" tab).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SampleRateLabel = Literal["32k", "40k", "48k"]
RvcVersion = Literal["v1", "v2"]


@dataclass
class TrainingParams:
    experiment_name: str = "my_voice"
    trainset_dir: str = "datasets/my_voice_wavs"
    sample_rate_label: SampleRateLabel = "40k"
    version: RvcVersion = "v2"
    if_f0: bool = True
    speaker_id: int = 0
    num_processes: int = 4
    f0_method: str = "rmvpe"
    gpus_for_rmvpe: str = "0"
    gpu_devices_train: str = "0"
    save_every_epoch: int = 5
    total_epochs: int = 200
    batch_size: int = 4
    save_only_latest: bool = True
    cache_dataset_in_gpu: bool = False
    save_weights_every_epoch: bool = False
    pretrained_g: str = ""
    pretrained_d: str = ""
    skip_index: bool = False
    extract_infer_pth: bool = False
    infer_weight_name: str = "my_voice_infer"
    g_checkpoint_for_extract: str = ""
    extract_info_str: str = "Extracted model."
    sr_dict: dict = field(
        default_factory=lambda: {"32k": 32000, "40k": 40000, "48k": 48000}
    )


def resolve_pretrained_paths(p: TrainingParams, path_suffix_v2: str) -> tuple[str, str]:
    import os

    f0_str = "f0" if p.if_f0 else ""
    sr = p.sample_rate_label

    def pth(kind: str) -> str:
        return f"assets/pretrained{path_suffix_v2}/{f0_str}{kind}{sr}.pth"

    pg, pd = pth("G"), pth("D")
    g = p.pretrained_g if p.pretrained_g else (pg if os.access(pg, os.F_OK) else "")
    d = p.pretrained_d if p.pretrained_d else (pd if os.access(pd, os.F_OK) else "")
    return g, d
