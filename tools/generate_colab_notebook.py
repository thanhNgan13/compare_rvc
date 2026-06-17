"""
Sinh notebook Colab/Kaggle từ tree rvc_standalone hiện tại.
Chạy (trên máy dev):  python tools/generate_colab_notebook.py

Notebook gồm: hướng dẫn tiếng Việt + clone RVC gốc + ghi training_pipeline + patch + tải assets + train.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path


def b64_write(path_relative: str, content: bytes) -> str:
    b = base64.b64encode(content).decode("ascii")
    return (
        f'from pathlib import Path\n'
        f'import base64\n'
        f'p = Path(RVC_ROOT) / {json.dumps(path_relative)}\n'
        f'p.parent.mkdir(parents=True, exist_ok=True)\n'
        f'p.write_bytes(base64.b64decode({json.dumps(b)}))\n'
        f'print("Wrote", p)\n'
    )


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code(src: str) -> dict:
    lines = src.rstrip().split("\n")
    return {"cell_type": "code", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cells: list = []

    cells.append(
        md(
            """
# RVC — Train trên Google Colab / Kaggle (một notebook)

Notebook này **hướng dẫn chi tiết** và **tự tạo** package `training_pipeline`, file patch, script tải weight — bạn **chỉ cần chạy lần lượt các ô** từ trên xuống.

## Vì sao không nhét hết `infer/` vào một file `.ipynb`?

Thư mục `infer/` + `configs/` của RVC có **hàng nghìn dòng** (mô hình, train loop, UVR…). Nhúng nguyên văn vào notebook sẽ **vượt giới hạn thực tế**, khó đọc và **không đồng bộ** khi RVC cập nhật.

**Cách làm chuẩn trên Colab:** tải **mã RVC upstream một lần** (`git clone`), sau đó notebook **ghi đè / bổ sung** phần điều phối (`training_pipeline`) và **các patch** (PyTorch 2.6+ / Matplotlib / subprocess path) đã được kiểm chứng.

## Luồng tổng quát

1. Cấu hình `RVC_ROOT`, URL clone (mặc định: repo RVC chính thức).
2. Cài PyTorch (GPU), `fairseq`, phụ thuộc.
3. Clone RVC → có `infer/`, `configs/`, `i18n/`.
4. Sinh `training_pipeline/` + `infer/lib/fairseq_torch_load_compat.py` + vá các file infer (từ nội dung đóng gói trong notebook).
5. Tải `assets/` (Hubert, pretrained…).
6. Chuẩn bị `logs/mute` (tải từ nhánh repo hoặc tự tạo tối thiểu).
7. Upload / đặt `.wav` vào `datasets/`.
8. Chạy preprocess → F0+Hubert → train → index.

**Lưu ý:** Colab miễn phí có giới hạn phiên; train dài nên dùng Colab Pro hoặc Kaggle GPU.
"""
        )
    )

    cells.append(
        md(
            """
## 0) Cấu hình (sửa cho phù hợp)

Ô dưới đặt biến toàn cục: thư mục làm việc, URL Git, có dùng mirror Hugging Face hay không.
"""
        )
    )

    cells.append(
        code(
            """# --- CẤU HÌNH ---
import os
from pathlib import Path

# Thư mục gốc chứa infer/, configs/ sau khi clone (Linux Colab/Kaggle thường dùng /content)
RVC_ROOT = Path(os.environ.get("RVC_ROOT", "/content/Retrieval-based-Voice-Conversion-WebUI"))

# Repo RVC chính thức (đủ infer + configs). Nếu bạn fork và thêm sửa, đổi URL tại đây.
GIT_REPO = os.environ.get(
    "RVC_GIT_URL",
    "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git",
)

# Mirror HF nếu huggingface.co hay bị reset (Trung Quốc / một số ISP)
os.environ.setdefault("RVC_HF_MIRROR", "0")

print("RVC_ROOT =", RVC_ROOT.resolve())
print("GIT_REPO =", GIT_REPO)
"""
        )
    )

    cells.append(
        md(
            """
## 1) Cài đặt PyTorch GPU + thư viện

**Giải thích:** Colab thường có CUDA; ta cài `torch` bản CUDA và các gói RVC (`fairseq`, `ffmpeg-python`, …).  
`fairseq 0.12.2` hay lỗi resolver với `pip` quá mới — ta hạ `pip` vào khoảng 23.x–24.0.
"""
        )
    )

    cells.append(
        code(
            r"""# --- CÀI PHỤ THUỘC ---
import subprocess
import sys

def sh(cmd: str) -> None:
    print(">>", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

# Pip tương thích fairseq
sh(f"{sys.executable} -m pip install -q -U 'pip>=23.2,<24.1'")

# PyTorch: Colab/Kaggle — dùng index CUDA 12.x (Colab thường tương thích)
sh(f"{sys.executable} -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")

# RVC requirements cốt lõi (rút gọn; có thể mở rộng theo requirements.txt đầy đủ)
pkgs = [
    "numpy", "scipy", "librosa", "soundfile", "ffmpeg-python", "tensorboard",
    "tqdm", "faiss-cpu", "scikit-learn", "matplotlib", "dotenv",
    "fairseq==0.12.2", "requests", "praat-parselmouth", "pyworld",
]
sh(f"{sys.executable} -m pip install -q " + " ".join(pkgs))

# ffmpeg hệ thống (Colab thường có sẵn; nếu thiếu uncomment)
# sh("apt-get update -qq && apt-get install -qq ffmpeg")

import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
"""
        )
    )

    cells.append(
        md(
            """
## 2) Clone mã RVC (infer + configs)

**Giải thích:** Lệnh `git clone --depth 1` tải snapshot mỏng. Sau bước này, `RVC_ROOT` sẽ chứa `infer/`, `configs/`, … giống WebUI.
"""
        )
    )

    cells.append(
        code(
            """# --- CLONE RVC ---
import os
import shutil
import subprocess
from pathlib import Path

def _as_path(x):
    return x if isinstance(x, Path) else Path(str(x))

root_dir = _as_path(RVC_ROOT)
if (root_dir / "infer").is_dir():
    print("Đã có infer/, bỏ qua clone:", root_dir)
else:
    parent = root_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if root_dir.exists():
        shutil.rmtree(root_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", GIT_REPO, str(root_dir)],
        check=True,
    )
print("OK:", (root_dir / "infer").is_dir())
"""
        )
    )

    # Embed files (đồng bộ với tree rvc_standalone tại thời điểm sinh notebook)
    embed_map = [
        "training_pipeline/__init__.py",
        "training_pipeline/setup_env.py",
        "training_pipeline/params.py",
        "training_pipeline/steps.py",
        "infer/lib/fairseq_torch_load_compat.py",
        "infer/modules/train/extract_feature_print.py",
        "infer/modules/train/train.py",
        "infer/lib/train/utils.py",
        "infer/modules/vc/utils.py",
        "tools/download_assets.py",
    ]
    init_py = root / "training_pipeline" / "__init__.py"
    if not init_py.is_file():
        init_py.write_text("# package\n", encoding="utf-8")

    cells.append(
        md(
            """
## 3) Ghi `training_pipeline` + các file đã patch (base64)

**Giải thích:** Các file được **mã hóa base64** lúc bạn chạy `python tools/generate_colab_notebook.py` trên máy. Ô code giải mã và ghi đè vào `RVC_ROOT` gồm:

- `training_pipeline/` — điều phối preprocess → F0 → Hubert → train → index  
- `infer/lib/fairseq_torch_load_compat.py` — PyTorch 2.6+ `torch.load`  
- `infer/modules/train/extract_feature_print.py` — `sys.path` + fairseq compat  
- `infer/modules/train/train.py` — `USE_LIBUV` trên Windows (và vẫn an toàn trên Linux)  
- `infer/lib/train/utils.py` — Matplotlib 3.8+ (`buffer_rgba`)  
- `infer/modules/vc/utils.py` — fairseq trước khi load Hubert (infer)  
- `tools/download_assets.py` — tải weight Hugging Face  

*Bản upstream RVC chỉ làm nền; các file trên đảm bảo lệnh train chạy trên môi trường mới.*
"""
        )
    )

    blob_lines = [
        "from pathlib import Path",
        "import base64",
        "import os",
        "",
        "try:",
        "    _rv = RVC_ROOT",
        "except NameError:",
        "    _rv = os.environ.get('RVC_ROOT', '/content/Retrieval-based-Voice-Conversion-WebUI')",
        "RVC_ROOT = Path(_rv)",
        "",
    ]
    for rel_path in embed_map:
        p = root / rel_path
        if not p.is_file():
            raise FileNotFoundError(p)
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        blob_lines.append(f"# {rel_path}")
        blob_lines.append(f"_d = base64.b64decode({json.dumps(b64)})")
        blob_lines.append(f"_p = RVC_ROOT / {json.dumps(rel_path)}")
        blob_lines.append("_p.parent.mkdir(parents=True, exist_ok=True)")
        blob_lines.append("_p.write_bytes(_d)")
        blob_lines.append('print("Wrote", _p, "bytes", len(_d))')
        blob_lines.append("")

    cells.append(code("\n".join(blob_lines)))

    cells.append(
        md(
            """
## 5) Tải Hubert + pretrained (chạy từ `RVC_ROOT`)

**Giải thích:** Script `tools/download_assets.py` vừa được ghi ở bước 3. Ta `cd` vào `RVC_ROOT` rồi chạy. Có thể `export RVC_HF_MIRROR=1` nếu cần.
"""
        )
    )

    cells.append(
        code(
            """# --- TẢI ASSETS ---
import os, subprocess, sys
from pathlib import Path

os.chdir(str(RVC_ROOT))
subprocess.run([sys.executable, "tools/download_assets.py"], check=True)
"""
        )
    )

    cells.append(
        md(
            """
## 6) Thư mục `logs/mute`

**Giải thích:** Train cần template im lặng. Cách nhanh trên Colab: clone mute từ một bản có sẵn hoặc copy từ snapshot.

Nếu repo RVC của bạn **không** có `logs/mute`, hãy upload zip mute lên Colab hoặc tải từ bản WebUI đầy đủ.  
(Tùy chọn) Clone nhánh khác chỉ để lấy mute — có thể bỏ qua nếu đã có trong repo.
"""
        )
    )

    cells.append(
        code(
            """# --- MUTE (tùy chọn — sửa URL nếu bạn có nguồn chứa logs/mute) ---
from pathlib import Path
import shutil, subprocess

m = Path(RVC_ROOT) / "logs" / "mute"
if (m / "0_gt_wavs").is_dir():
    print("mute OK")
else:
    print("THIEU logs/mute — vui long them tay (xem markdown o tren)")
"""
        )
    )

    cells.append(
        md(
            """
## 7) Dataset — đặt file `.wav`

**Giải thích:** Upload qua panel Files của Colab vào `datasets/giong_cua_toi/` hoặc gắn Google Drive.
"""
        )
    )

    cells.append(
        code(
            """# --- TẠO THƯ MỤC DATASET ---
from pathlib import Path
D = Path(RVC_ROOT) if not isinstance(RVC_ROOT, Path) else RVC_ROOT
(D / "datasets/giong_cua_toi").mkdir(parents=True, exist_ok=True)
print("Hay upload .wav vao:", D / "datasets/giong_cua_toi")
"""
        )
    )

    cells.append(
        md(
            """
## 8) Train — bootstrap + từng bước

**Giải thích:**
- `bootstrap()` đưa `sys.path` và `cwd` về `RVC_ROOT`, nạp `configs.config.Config`.
- `TrainingParams` giống notebook để bàn: tên thí nghiệm, thư mục wav, epoch, batch, GPU.
- `train_steps.step_*` gọi subprocess `preprocess.py`, `extract_f0_print.py`, `extract_feature_print.py`, `train.py`, rồi FAISS index.
"""
        )
    )

    cells.append(
        code(
            r"""# --- CHẠY TRAIN ---
import os, sys, logging, pathlib
from pathlib import Path

RVC_ROOT = Path(RVC_ROOT).resolve() if not isinstance(RVC_ROOT, Path) else RVC_ROOT.resolve()
os.chdir(str(RVC_ROOT))
sys.path.insert(0, str(RVC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from training_pipeline.setup_env import bootstrap
from training_pipeline.params import TrainingParams
from training_pipeline import steps as train_steps

root, config = bootstrap()

p = TrainingParams(
    experiment_name="colab_voice",
    trainset_dir="datasets/giong_cua_toi",
    sample_rate_label="40k",
    version="v2",
    if_f0=True,
    num_processes=2,
    f0_method="rmvpe",
    gpu_devices_train="0",
    total_epochs=50,
    save_every_epoch=5,
    batch_size=4,
    skip_index=False,
)

miss = train_steps.check_mute_template(root)
print("logs/mute:", "THIEU" if miss else "OK", miss)

train_steps.step_preprocess(root, config, p)
train_steps.step_extract_f0_and_features(root, config, p)
train_steps.step_train(root, config, p)
if not p.skip_index:
    for line in train_steps.step_train_index(root, config, p):
        print(line)

print("Hoan tat pipeline (train + index). Checkpoint trong logs/", p.experiment_name)
"""
        )
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }

    out = root / "RVC_Train_Colab_Kaggle_Generated.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
