"""
Tải pretrained / Hubert / RMVPE / UVR5 vào assets/ của gói rvc_standalone.

Chạy từ bất kỳ đâu:
  cd /path/to/rvc_standalone
  python tools/download_assets.py

Cần: pip install requests

Lỗi ConnectionResetError (WinError 10054) / Connection aborted:
  - Mạng / firewall / VPN / nhà mạng đóng kết nối HTTPS tới Hugging Face.
  - Chạy lại script (đã có retry tự động); thử VPN ổn định hoặc mạng khác.
  - Thử mirror (một số vùng mạng ổn định hơn):
      set RVC_HF_MIRROR=1
      python tools/download_assets.py
    (PowerShell: $env:RVC_HF_MIRROR="1"; python tools/download_assets.py)
  - Hoặc tải tay: https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

# Thư mục gốc = cha của tools/ (chính là rvc_standalone)
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_HF = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
MIRROR_HF = "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/"


def _base_url() -> str:
    if os.environ.get("RVC_HF_BASE"):
        return os.environ["RVC_HF_BASE"].rstrip("/") + "/"
    if os.environ.get("RVC_HF_MIRROR", "").lower() in ("1", "true", "yes"):
        return MIRROR_HF
    return DEFAULT_HF


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; RVC-standalone-download/1.1)",
        }
    )
    return s


def dl_model(
    link: str,
    model_name: str,
    dir_name: Path,
    *,
    session: requests.Session,
    retries: int = 6,
    connect_timeout: int = 60,
    read_timeout: int = 600,
) -> None:
    url = f"{link}{model_name}"
    out = dir_name / model_name.split("/")[-1]
    out.parent.mkdir(parents=True, exist_ok=True)
    timeout = (connect_timeout, read_timeout)
    last_err: Exception | None = None
    for attempt in range(retries):
        tmp: Path | None = None
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = out.with_suffix(out.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                tmp.replace(out)
            return
        except (requests.RequestException, ConnectionError, OSError) as e:
            last_err = e
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            wait = min(8 * (2**attempt), 120)
            print(
                f"  [!] Lỗi tải (lần {attempt + 1}/{retries}): {e}\n"
                f"      Chờ {wait}s rồi thử lại..."
            )
            time.sleep(wait)
    raise RuntimeError(f"Tải thất bại sau {retries} lần: {url}\nGốc lỗi: {last_err}") from last_err


if __name__ == "__main__":
    base = _base_url()
    print("BASE_DIR =", BASE_DIR)
    print("HF base URL =", base)
    if base == MIRROR_HF:
        print("(Đang dùng hf-mirror; đặt RVC_HF_MIRROR=0 hoặc xóa biến để dùng huggingface.co)")

    sess = _session()

    print("Downloading hubert_base.pt...")
    dl_model(base, "hubert_base.pt", BASE_DIR / "assets/hubert", session=sess)
    print("Downloading rmvpe.pt...")
    dl_model(base, "rmvpe.pt", BASE_DIR / "assets/rmvpe", session=sess)
    print("Downloading vocals.onnx...")
    dl_model(
        base + "uvr5_weights/onnx_dereverb_By_FoxJoy/",
        "vocals.onnx",
        BASE_DIR / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy",
        session=sess,
    )

    rvc_models_dir = BASE_DIR / "assets/pretrained"
    print("Downloading pretrained models (v1 folder)...")
    model_names = [
        "D32k.pth",
        "D40k.pth",
        "D48k.pth",
        "G32k.pth",
        "G40k.pth",
        "G48k.pth",
        "f0D32k.pth",
        "f0D40k.pth",
        "f0D48k.pth",
        "f0G32k.pth",
        "f0G40k.pth",
        "f0G48k.pth",
    ]
    for model in model_names:
        print(f"  {model}...")
        dl_model(base + "pretrained/", model, rvc_models_dir, session=sess)

    rvc_models_dir = BASE_DIR / "assets/pretrained_v2"
    print("Downloading pretrained_v2...")
    for model in model_names:
        print(f"  {model}...")
        dl_model(base + "pretrained_v2/", model, rvc_models_dir, session=sess)

    rvc_models_dir = BASE_DIR / "assets/uvr5_weights"
    print("Downloading uvr5_weights...")
    uvr_names = [
        "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]
    for model in uvr_names:
        print(f"  {model}...")
        dl_model(base + "uvr5_weights/", model, rvc_models_dir, session=sess)

    print("Done. Kiểm tra assets/hubert, assets/pretrained(_v2), assets/rmvpe, assets/uvr5_weights.")
