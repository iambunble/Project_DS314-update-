# Tạo môi trường ảo .venv và cài thư viện cho dự án.
# Chạy trong terminal PowerShell của VS Code:   .\setup_venv.ps1
# Có card NVIDIA:                               .\setup_venv.ps1 -Gpu
param([switch]$Gpu)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Tao moi truong ao .venv ..." -ForegroundColor Cyan
    py -3 -m venv .venv 2>$null
    if (-not (Test-Path ".venv")) { python -m venv .venv }
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

if ($Gpu) {
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu121
} else {
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install -r requirements.txt

Write-Host "`nXong! Kich hoat moi truong:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Sau do build chi muc:         python scripts\build_index.py" -ForegroundColor Green
Write-Host "Mo giao dien:                 streamlit run app\streamlit_app.py" -ForegroundColor Green
