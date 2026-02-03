@echo off
cd /d "%~dp0"
echo VoiceShield - Starting web app...
echo.
echo After startup, open in your browser:  http://localhost:8000
echo.
python run_web.py 2>nul || py run_web.py 2>nul || python3 run_web.py
if errorlevel 1 (
  echo.
  echo Python not found or dependencies missing. From this folder run:
  echo   pip install -r requirements.txt
  echo   python run_web.py
  pause
)
