@echo off
REM LLMlifeline startup script
REM Launches the backend using the venv's python.exe directly, by full path.
REM This bypasses PATH/activation entirely, sidestepping a batch-script
REM quirk where PATH changes from activate.bat don't always propagate
REM correctly within the same script execution.
REM
REM One-click flow: this script bootstraps a completely fresh machine —
REM creates the Python virtual environment and clones ComfyUI if either
REM is missing (see the two bootstrap sections below) — installs any
REM missing Python packages, starts the backend in its own window, waits
REM for it to actually become reachable, then opens your default browser
REM to it automatically. The backend also serves the frontend page itself
REM (see app.mount in backend/main.py), so there's exactly one URL to
REM open — no separate frontend/index.html file to open by hand.
REM
REM This is what makes handing the whole project folder (or a zip of it)
REM to someone else's PC actually work: they don't need to know what a
REM virtual environment or ComfyUI even is. What they DO still need
REM before this can succeed: Python 3.10 installed, git installed, an
REM NVIDIA GPU with a current driver, and their own model files dropped
REM into models\ and ComfyUI\models\checkpoints\ — none of that can be
REM bundled or automated away (see setup-guide.md for why).

cd /d "%~dp0"

if not exist "backend\main.py" (
    echo ERROR: Could not find backend\main.py — is this script sitting in
    echo the LLMlifeline project root?
    pause
    exit /b 1
)

set VENV_PYTHON=%CD%\llm-env\Scripts\python.exe

REM ---- Bootstrap 1: create llm-env from scratch if missing or broken ----
REM A venv bakes in an absolute path to its base Python install (see
REM llm-env\pyvenv.cfg) and simply does not run on a different machine —
REM confirmed directly: copying this project elsewhere leaves a
REM llm-env\Scripts\python.exe file that LOOKS present but fails the
REM moment it's actually invoked. "exists" alone isn't a good enough
REM check, so this actually tries to run it before trusting it.
set NEED_FRESH_VENV=0
if not exist "%VENV_PYTHON%" set NEED_FRESH_VENV=1
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -c "import sys" >nul 2>nul
    if errorlevel 1 set NEED_FRESH_VENV=1
)

if %NEED_FRESH_VENV%==0 goto venv_ready

echo No working Python virtual environment found - setting one up now.
echo This is a one-time step.
if exist "llm-env" (
    echo   Removing the existing llm-env folder first ^(it looks broken,
    echo   most likely copied from a different machine — venvs aren't
    echo   portable^)...
    rmdir /s /q "llm-env"
)

set SYSTEM_PYTHON=
py -3.10 --version >nul 2>nul
if not errorlevel 1 set SYSTEM_PYTHON=py -3.10
if "%SYSTEM_PYTHON%"=="" (
    python --version >nul 2>nul
    if not errorlevel 1 set SYSTEM_PYTHON=python
)
if "%SYSTEM_PYTHON%"=="python" (
    echo   NOTE: Python 3.10 specifically wasn't found - using whatever
    echo   "python" resolves to on PATH instead. This project was built and
    echo   tested against 3.10; most dependencies publish wheels for a wide
    echo   version range, so this will often still work, but if package
    echo   installs fail below, installing Python 3.10 itself
    echo   ^(https://www.python.org/downloads/^) is the safest fix.
)
if "%SYSTEM_PYTHON%"=="" (
    echo ERROR: No Python installation found on this machine.
    echo Install Python 3.10 from https://www.python.org/downloads/
    echo ^(check "Add python.exe to PATH" during install^), then run this
    echo script again.
    pause
    exit /b 1
)

%SYSTEM_PYTHON% -m venv llm-env
if errorlevel 1 (
    echo ERROR: Could not create the virtual environment. See output above.
    pause
    exit /b 1
)
echo   Virtual environment created.
echo.

:venv_ready
echo Using Python: %VENV_PYTHON%
echo.

REM ---- Bootstrap 2: clone ComfyUI fresh if it's missing ----
REM ComfyUI is a real upstream project (github.com/comfyanonymous/ComfyUI),
REM not code that belongs to this one, so it's never copied/zipped along
REM with the rest of this project on purpose (see setup-guide.md) — it
REM gets cloned here instead, pinned to the exact commit this project's
REM image-generation code (backend/main.py's txt2img workflow + API
REM calls) was actually built and tested against, rather than tracking
REM upstream HEAD — a breaking change there shouldn't silently break
REM Image Model here.
if not exist "ComfyUI\main.py" (
    echo ComfyUI not found - cloning it fresh from GitHub. This is a real
    echo ~6-7GB download and can take a while on a slow connection.
    where git >nul 2>nul
    if errorlevel 1 (
        echo ERROR: git is not installed or not on PATH.
        echo Install it from https://git-scm.com/downloads, then run this
        echo script again.
        pause
        exit /b 1
    )
    if exist "ComfyUI" (
        echo   Removing incomplete ComfyUI folder...
        rmdir /s /q "ComfyUI"
    )
    git clone https://github.com/comfyanonymous/ComfyUI.git
    if errorlevel 1 (
        echo ERROR: Could not clone ComfyUI. See output above.
        pause
        exit /b 1
    )
    pushd ComfyUI
    git checkout 8a33128f2f8c5585c57486c07de481241e70a39c >nul 2>nul
    popd
    echo   ComfyUI cloned.
    echo.
)

REM ---- Dependency check ----
REM Checks each required package individually rather than a single
REM `pip install -r requirements.txt` call, so a person only sees output
REM for what's actually missing, and an already-complete environment
REM starts up fast with no wasted pip calls. Uses `python -m pip`, not
REM the standalone pip.exe launcher, which has been confirmed broken on
REM this venv after being moved (its launcher header hardcodes the old
REM path) - `-m pip` bypasses that entirely.
REM
REM NOTE on llama-cpp-python specifically: this checks that it's
REM importable, but does NOT verify it was built with CUDA support. A
REM fresh `pip install llama-cpp-python` with no CMAKE_ARGS set installs
REM a CPU-only build, which will run but silently lose GPU acceleration -
REM it will NOT error or warn you. If this script had to install
REM llama-cpp-python fresh (see message below), rebuilding it with CUDA
REM per the original setup process is a separate, necessary step. See
REM setup-guide.md for what that involves.
echo Checking dependencies...
set NEEDS_INSTALL=0

"%VENV_PYTHON%" -c "import fastapi" 2>nul
if errorlevel 1 (
    echo   Missing: fastapi
    set NEEDS_INSTALL=1
)
"%VENV_PYTHON%" -c "import uvicorn" 2>nul
if errorlevel 1 (
    echo   Missing: uvicorn
    set NEEDS_INSTALL=1
)
"%VENV_PYTHON%" -c "import httpx" 2>nul
if errorlevel 1 (
    echo   Missing: httpx
    set NEEDS_INSTALL=1
)
"%VENV_PYTHON%" -c "import llama_cpp" 2>nul
if errorlevel 1 (
    echo   Missing: llama-cpp-python
    set NEEDS_INSTALL=1
)
"%VENV_PYTHON%" -c "import pypdf" 2>nul
if errorlevel 1 (
    echo   Missing: pypdf
    set NEEDS_INSTALL=1
)
"%VENV_PYTHON%" -c "import multipart" 2>nul
if errorlevel 1 (
    echo   Missing: python-multipart (needed for image/document uploads)
    set NEEDS_INSTALL=1
)
"%VENV_PYTHON%" -c "import transformers, PIL" 2>nul
if errorlevel 1 (
    echo   Missing: transformers/Pillow (needed for image captioning)
    set NEEDS_INSTALL=1
)
"%VENV_PYTHON%" -c "import comfy_kitchen" 2>nul
if errorlevel 1 (
    echo   Missing: comfy_kitchen
    set NEEDS_INSTALL=1
)

if %NEEDS_INSTALL%==1 (
    echo.
    echo Installing missing packages. This is a one-time step and may
    echo take several minutes, especially for llama-cpp-python.
    echo.
    "%VENV_PYTHON%" -m pip install --break-system-packages -r "%CD%\backend\requirements.txt"
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency install failed. See output above.
        pause
        exit /b 1
    )
    echo.
    echo NOTE: if llama-cpp-python was just installed fresh above, it is
    echo a CPU-only build by default and will run WITHOUT GPU
    echo acceleration. Rebuilding it with CUDA support requires the
    echo original CMAKE_ARGS build process, not this script — see setup-guide.md.
    echo.
)

REM ---- Torch / CUDA check (self-healing, not just a warning) ----
REM Torch is handled separately from the generic requirements install
REM above, on purpose: a plain `pip install torch` (what a bare
REM requirements.txt entry would trigger) resolves to a CPU-only build
REM from PyPI's default index. CUDA-enabled torch only exists on
REM PyTorch's own separate package index, which requires a distinct
REM --index-url flag that a requirements.txt file can't cleanly express
REM alongside other plain-PyPI packages.
REM
REM Also checks the torch VERSION, not just CUDA availability: comfy_kitchen
REM (ComfyUI's kernel library, pinned in backend/requirements.txt) uses
REM custom-op type hints that torch's own schema inference only accepts
REM from 2.7.0 onward — an older CUDA-enabled torch (e.g. a leftover
REM 2.5.1+cu121 install) still "has CUDA" but crashes ComfyUI on import.
REM Confirmed and fixed on this exact setup: reinstalling from
REM https://download.pytorch.org/whl/cu126 (2.7.1) resolved it.
echo Checking torch/CUDA status...
"%VENV_PYTHON%" -c "import torch; from packaging.version import Version; exit(0 if torch.cuda.is_available() and Version(torch.__version__.split('+')[0]) >= Version('2.7.0') else 1)" 2>nul
if errorlevel 1 (
    echo   torch is missing, CPU-only, or older than the 2.7.0 comfy_kitchen
    echo   needs. Installing a current CUDA-enabled build from PyTorch's
    echo   cu126 index. This is a large download and may take several minutes.
    "%VENV_PYTHON%" -m pip uninstall torch torchvision torchaudio -y >nul 2>nul
    "%VENV_PYTHON%" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 (
        echo.
        echo ERROR: torch CUDA install failed. See output above.
        echo ComfyUI image generation will not have GPU acceleration
        echo until this is resolved.
        pause
    ) else (
        "%VENV_PYTHON%" -c "import torch; from packaging.version import Version; exit(0 if torch.cuda.is_available() and Version(torch.__version__.split('+')[0]) >= Version('2.7.0') else 1)" 2>nul
        if errorlevel 1 (
            echo   WARNING: torch reinstalled but still does not report CUDA
            echo   availability. This may indicate a driver or CUDA toolkit
            echo   mismatch beyond what this script can fix automatically.
        ) else (
            echo   torch CUDA install succeeded.
        )
    )
) else (
    echo   torch CUDA: OK
)
echo.

REM ---- Default model download (first run only, ever) ----
REM GGUF/safetensors files are too large for GitHub, so instead of
REM shipping them, download_models.py fetches this project's default
REM models the first time the app is ever started. It gates itself on
REM its own marker file (models\.default_models_downloaded) and is a
REM fast no-op on every later run — see that script for the full
REM reasoning, including why it deliberately never re-downloads
REM something someone removed on purpose.
"%VENV_PYTHON%" download_models.py

REM ---- Model files sanity check ----
REM Belt-and-suspenders alongside download_models.py above — catches
REM the case where that step was skipped, failed partway, or someone
REM removed everything afterward, so the best this script can still do
REM is tell someone clearly that they're missing, instead of leaving
REM them to wonder why the model dropdown is empty once the app opens.
set HAS_MODELS=0
if exist "models\*.gguf" set HAS_MODELS=1
if %HAS_MODELS%==0 (
    echo NOTE: No .gguf chat models found in models\ yet. Add at least one
    echo there before trying to chat - see setup-guide.md.
    echo.
)
set HAS_CHECKPOINTS=0
if exist "ComfyUI\models\checkpoints\*.safetensors" set HAS_CHECKPOINTS=1
if %HAS_CHECKPOINTS%==0 (
    echo NOTE: No image checkpoints found in ComfyUI\models\checkpoints\ yet.
    echo Add at least one .safetensors file there before trying Image Model.
    echo.
)

REM ---- Launch + auto-open browser ----
REM Runs uvicorn in its own minimized window rather than blocking this
REM one, so this script can poll for readiness and then open the browser
REM itself — this is what makes double-clicking start.bat a genuinely
REM complete one-click flow instead of "start the backend, then go open
REM the page yourself."
echo Starting backend on port 8001...
start "LLMlifeline Backend" /min cmd /c ""%VENV_PYTHON%" -m uvicorn main:app --app-dir backend --port 8001"

echo Waiting for the backend to become ready (this can take a while on a
echo cold model load)...
REM Uses curl.exe (built into Windows since the 1803 update, a real
REM binary at C:\Windows\system32\curl.exe — not a PowerShell alias) for
REM the readiness check, and `ping -n 2 127.0.0.1` as the ~1-second wait
REM between attempts. Both were chosen specifically over `powershell
REM -Command ...` and `timeout /nobreak` — both of those depend on a
REM real interactive console being attached and fail with "Input
REM redirection is not supported" when it isn't (confirmed while testing
REM this exact script). ping/curl have no such dependency.
setlocal enabledelayedexpansion
set READY=0
for /l %%i in (1,1,180) do (
    if !READY!==0 (
        curl.exe -s -f -m 2 -o nul http://127.0.0.1:8001/api/health
        if not errorlevel 1 set READY=1
    )
    if !READY!==0 ping -n 2 127.0.0.1 >nul
)

if !READY!==1 (
    echo Backend is up — opening the browser...
    start http://127.0.0.1:8001/
) else (
    echo.
    echo The backend did not respond within 3 minutes. It may still be
    echo loading a large model — check the "LLMlifeline Backend" window,
    echo or open http://127.0.0.1:8001/ yourself once it settles.
)

echo.
echo The backend is running in a separate minimized window titled
echo "LLMlifeline Backend". Closing THIS window is fine — use the Stop
echo button in the browser page when you're done, or run stop.bat.
echo.
pause
