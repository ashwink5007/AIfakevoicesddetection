"""
Start VoiceShield web app. Run from ml-service:
  python run_web.py
  or: py run_web.py
Then open: http://localhost:8000
"""
import sys
import os
import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
