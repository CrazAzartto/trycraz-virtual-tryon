@echo off
REM Double-click to start the app.
REM It asks for the server URL and API key printed by the Colab/Kaggle notebook (step 4)
REM and opens the app in your browser with both already filled in.
REM Leave both empty to use only the Recommendations page (no GPU needed).

cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
    echo The Python environment .venv was not found. See README.md, section "Setup".
    pause
    exit /b 1
)

echo.
echo === Trycraz ===
echo Paste the URL printed by the notebook (looks like https://xxxx.trycloudflare.com)
echo and press Enter. Right-click pastes in this window.
set /p TRYON_SERVER_URL=Server URL:
echo.
echo Paste the API key printed by the notebook and press Enter.
set /p TRYON_API_KEY=API key:
echo.
echo Starting Trycraz... it will open in your browser at http://localhost:8501
echo Keep this window open while you use the app. Close it to stop the app.
echo.
".venv\Scripts\streamlit.exe" run app.py
pause
