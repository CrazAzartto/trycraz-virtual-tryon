@echo off
setlocal
REM ==========================================================================
REM  Trycraz launcher for Windows. Double-click this file to start the app.
REM
REM  First run: it creates the Python environment (.venv) and installs the
REM  libraries from requirements.lock.txt. This happens only once.
REM  Every run: it asks for the server URL and API key printed by the Kaggle
REM  or Colab notebook (step 4) and opens the app in your browser.
REM  Leave both empty to use only the Recommendations page (no GPU needed).
REM
REM  start_app.bat --setup-only   sets up .venv and exits without starting.
REM  Full instructions: docs\SETUP.md
REM ==========================================================================

cd /d "%~dp0"
if exist ".venv\Scripts\streamlit.exe" goto run

echo.
echo === Trycraz: first-time setup ===
echo The Python environment (.venv) is not set up yet. Setting it up downloads
echo and installs the required libraries (about 1 to 2 GB). It takes about
echo 5 to 15 minutes and is needed only once.
echo.
set "GO=Y"
set /p "GO=Set it up now? [Y/n] "
if /i "%GO%"=="n" (
    echo Setup cancelled. See docs\SETUP.md for manual installation.
    pause
    exit /b 1
)

REM --- Option 1: uv (fast; can also download Python 3.12 by itself) ---
where uv >nul 2>&1
if errorlevel 1 goto use_pip
echo.
echo Using uv to create .venv with Python 3.12 ...
if exist ".venv" rmdir /s /q ".venv"
uv venv .venv --python 3.12
if errorlevel 1 goto use_pip
uv pip install --python ".venv\Scripts\python.exe" -r requirements.lock.txt
if errorlevel 1 (
    echo The exact versions could not be installed. Trying requirements.txt instead ...
    uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
)
goto check_setup

REM --- Option 2: the Python launcher or python on PATH, with pip ---
:use_pip
set "PYEXE="
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3.12"
if defined PYEXE goto make_venv
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3.11"
if defined PYEXE goto make_venv
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 set "PYEXE=python"
if defined PYEXE goto make_venv
echo.
echo Python 3.12 was not found on this computer.
echo Install Python 3.12 from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" in the installer), then run this file again.
echo See docs\SETUP.md for details.
pause
exit /b 1

:make_venv
echo.
echo Creating .venv with %PYEXE% ...
if exist ".venv" rmdir /s /q ".venv"
%PYEXE% -m venv .venv
if errorlevel 1 goto setup_failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.lock.txt
if errorlevel 1 (
    echo The exact versions could not be installed. Trying requirements.txt instead ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

:check_setup
if exist ".venv\Scripts\streamlit.exe" (
    echo.
    echo Setup finished.
    if /i "%~1"=="--setup-only" exit /b 0
    goto run
)

:setup_failed
echo.
echo Setup did not finish. Check your internet connection and run this file again.
echo If it keeps failing, follow the manual steps in docs\SETUP.md.
pause
exit /b 1

:run
if /i "%~1"=="--setup-only" exit /b 0
echo.
echo === Trycraz ===
echo Paste the URL printed by the notebook (looks like https://xxxx.trycloudflare.com)
echo and press Enter. Right-click pastes in this window.
echo Leave it empty and press Enter to use only the Recommendations page.
set "TRYON_SERVER_URL="
set /p "TRYON_SERVER_URL=Server URL: "
echo.
echo Paste the API key printed by the notebook and press Enter.
set "TRYON_API_KEY="
set /p "TRYON_API_KEY=API key: "
echo.
echo Starting Trycraz... it will open in your browser at http://localhost:8501
echo Keep this window open while you use the app. Close it to stop the app.
echo.
".venv\Scripts\streamlit.exe" run app.py
pause
