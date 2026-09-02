@echo off
REM LLMlifeline shutdown script
REM Uses PowerShell's Get-NetTCPConnection (structured data, not text
REM parsing) to reliably find and kill whatever's listening on ports
REM 8000 (llama_cpp.server), 8001 (backend), and 8188 (ComfyUI).
REM
REM This is the manual fallback — the in-app Stop button (POST
REM /api/shutdown) already does this cleanly through the backend itself.
REM Use this script when that's not reachable: a crashed backend, a
REM closed browser tab, or a hung process from a previous session.

echo Stopping LLMlifeline processes...
echo.

powershell -NoProfile -Command ^
    "$ports = 8000, 8001, 8188; " ^
    "$found = $false; " ^
    "foreach ($p in $ports) { " ^
    "  $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; " ^
    "  foreach ($c in $conns) { " ^
    "    $found = $true; " ^
    "    Write-Host \"Killing process on port $p (PID $($c.OwningProcess))...\"; " ^
    "    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; " ^
    "  } " ^
    "}; " ^
    "if (-not $found) { Write-Host 'Nothing was listening on ports 8000 or 8001.' }; " ^
    "Start-Sleep -Seconds 1; " ^
    "Write-Host ''; " ^
    "Write-Host 'Verifying ports are clear...'; " ^
    "foreach ($p in $ports) { " ^
    "  $stillThere = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; " ^
    "  if ($stillThere) { Write-Host \"Port $p : WARNING - still in use\" } else { Write-Host \"Port $p : clear\" } " ^
    "}"

echo.
pause
