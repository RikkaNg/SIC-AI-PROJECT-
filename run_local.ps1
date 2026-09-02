# ============================================================
# run_local.ps1 - Khoi dong toan bo SIC-AI-PROJECT KHONG Docker
# Mo 3 cua so PowerShell rieng: ML Service / Backend / Frontend
# Chay:  powershell -ExecutionPolicy Bypass -File run_local.ps1
# ============================================================

$root = Split-Path -Parent $MyInvocation.MyCommand.Path   # thu muc goc du an

# --- Ham nạp bien moi truong tu .env vao process hien tai ---
$envLoader = @'
Get-Content '.\.env' -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        Set-Item -Path "env:$($Matches[1].Trim())" -Value $Matches[2].Trim()
    }
}
'@

Write-Host "== 0/3 Sinh JWT_SECRET neu con la gia tri dev mac dinh ==" -ForegroundColor Cyan
python (Join-Path $root 'backend\scripts\gen_jwt_secret.py')

Write-Host "== 1/3 ML Service  -> http://localhost:8001 ==" -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit', '-Command', @"
Set-Location '$root\ml_service\app'
python -m uvicorn main:app --host 127.0.0.1 --port 8001
"@

Write-Host "== 2/3 Backend Gateway -> http://localhost:8000 ==" -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit', '-Command', @"
$envLoader
Set-Location '$root'
python -m uvicorn backend.src.main:app --host 127.0.0.1 --port 8000
"@

Write-Host "== 3/3 Frontend (Vite) -> http://localhost:5173 ==" -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit', '-Command', @"
Set-Location '$root\frontend'
if (-not (Test-Path node_modules)) { npm install }
npm run dev
"@

Write-Host ""
Write-Host "Da mo 3 cua so. Cho ~20-40s roi mo:  http://localhost:5173" -ForegroundColor Green
Write-Host "Tai khoan: admin/admin123 | manager1/manager123 | manager2/manager123"
