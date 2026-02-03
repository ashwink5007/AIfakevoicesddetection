"""
VoiceShield - Prediction module.
Preprocess → extract features → load model → classify REAL or FAKE with confidence.
"""

import pickle
import numpy as np
from pathlib import Path

from .preprocess import load_and_preprocess
from .extract_features import extract_features_from_waveform, TARGET_SR

MODEL_SAVE_PATH = "models/voice_model.pkl"


def _load_artifact(model_path: str) -> dict:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(audio_path: str, model_path: str = MODEL_SAVE_PATH) -> tuple[str, float]:
    """
    Run VoiceShield on a single audio file.

    Args:
        audio_path: Path to the audio file.
        model_path: Path to the saved model pickle.

    Returns:
        (label, confidence): "REAL" or "FAKE", and confidence in [0, 1].
    """
    artifact = _load_artifact(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    waveform = load_and_preprocess(audio_path)
    feature_vector = extract_features_from_waveform(waveform, TARGET_SR)
    # Ensure same order as training
    X = np.array([feature_vector])  # shape (1, n_features)
    if X.shape[1] != len(feature_columns):
        raise ValueError(
            f"Feature dimension mismatch: got {X.shape[1]}, expected {len(feature_columns)}. "
            "Retrain the model with the current extract_features pipeline."
        )
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]  # 1 = inlier (REAL), -1 = outlier (FAKE)
    # Confidence from decision function (higher = more "normal")
    decision = model.decision_function(X_scaled)[0]
    # Map to [0, 1]: shift and scale (decision range is dataset-dependent)
    confidence = _decision_to_confidence(decision, model)
    label = "REAL" if prediction == 1 else "FAKE"
    return label, confidence


def _decision_to_confidence(decision: float, model) -> float:
    """
    Map Isolation Forest decision_function to a 0–1 confidence score.
    Higher decision = more inlier-like = higher confidence for REAL.
    """
    # decision_function: negative = more anomalous. We want REAL → high confidence.
    # Simple sigmoid-like scaling: clip and normalize to [0, 1]
    try:
        # Typical range is roughly [-0.5, 0.5] for sklearn's Isolation Forest
        shifted = decision + 0.5
        confidence = np.clip(shifted, 0, 1)
        return float(confidence)
    except Exception:
        return 0.5


def predict_and_print(audio_path: str, model_path: str = MODEL_SAVE_PATH) -> None:
    """
    Run prediction and print result in the required format.
    Example output:
      Prediction: REAL
      Confidence: 91%
    """
    label, confidence = predict(audio_path, model_path)
    pct = int(round(confidence * 100))
    print(f"Prediction: {label}")
    print(f"Confidence: {pct}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_path> [model_path]")
        sys.exit(1)
    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_SAVE_PATH
    predict_and_print(audio_path, model_path)
