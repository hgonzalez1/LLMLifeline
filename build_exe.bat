@echo off
REM Builds dist\LLMlifeline.exe from launcher.py via PyInstaller.
REM
REM Re-run this after any change to launcher.py to regenerate the exe —
REM it's not a one-off artifact, it's meant to be rebuilt as the project
REM changes. Uses the project's own venv (llm-env), so PyInstaller only
REM needs to be installed there once:
REM   llm-env\Scripts\python.exe -m pip install pyinstaller
REM
REM This does NOT bundle torch/CUDA/ComfyUI/model weights into the exe —
REM see launcher.py's own docstring for why. The exe is a small, fast
REM orchestrator; the real dependencies still live in llm-env and get
REM installed there on first run, same as start.bat.

cd /d "%~dp0"

set VENV_PYTHON=%CD%\llm-env\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo ERROR: Could not find venv Python at:
    echo %VENV_PYTHON%
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller into llm-env, one-time...
    "%VENV_PYTHON%" -m pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Could not install PyInstaller. See output above.
        pause
        exit /b 1
    )
)

echo Building LLMlifeline.exe...
echo.

REM --console: kept deliberately, not --windowed. First-run dependency
REM installs and cold model loads take minutes; a non-technical user
REM needs to see something happening, not a window that looks frozen.
"%VENV_PYTHON%" -m PyInstaller --onefile --console --name LLMlifeline ^
    --distpath "%CD%" --workpath "%CD%\build" --specpath "%CD%\build" ^
    launcher.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed. See output above.
    pause
    exit /b 1
)

echo.
echo Done: %CD%\LLMlifeline.exe
echo.
echo NOTE: this exe is unsigned. Windows SmartScreen/Defender will likely
echo flag it the first time it's run on a machine that didn't build it —
echo that's expected for any new, unsigned binary, not a sign of a
echo problem. "More info" -^> "Run anyway" gets past it.
echo.
pause
