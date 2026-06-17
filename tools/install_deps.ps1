# fairseq 0.12.2 + omegaconf 2.0.x: can pip < 24.1 de pip khong bo qua wheel omegaconf cu.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "[rvc_standalone] Downgrading pip to <24.1 for fairseq resolver..."
python -m pip install -U "pip>=23.2,<24.1"

Write-Host "[rvc_standalone] pip version:" (python -m pip --version)
Write-Host "[rvc_standalone] Installing requirements.txt..."
pip install -r (Join-Path $Root "requirements.txt")

Write-Host "[rvc_standalone] Done. Neu dung GPU NVIDIA, cai them PyTorch CUDA, vi du:"
Write-Host '  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124'
