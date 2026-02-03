"""
VoiceShield - End-to-end pipeline runner.
Run from ml-service directory: python run_pipeline.py [data_dir]
  - Extracts features from data_dir (default: data)
  - Trains model and saves to models/voice_model.pkl
Then run prediction: python src/predict.py <audio_path>
"""

import sys
import os

# Ensure ml-service is on path when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.extract_features import extract_features, FEATURES_CSV
from src.train_model import train_model, MODEL_SAVE_PATH

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    print("Step 1: Extracting features from real voice data...")
    extract_features(data_dir, FEATURES_CSV)
    print("Step 2: Training one-class model...")
    train_model(FEATURES_CSV, MODEL_SAVE_PATH)
    print("Pipeline complete. Run: python src/predict.py <audio_path>")
