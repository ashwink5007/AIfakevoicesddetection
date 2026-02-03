"""
VoiceShield - Model training module.
One-class learning on real voice features only; saves model and scaler for prediction.
"""

import pickle
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = "features/features.csv"
MODEL_SAVE_PATH = "models/voice_model.pkl"
FEATURE_COLUMNS = None  # Set from CSV (all numeric except filepath)


def _get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"filepath"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def train_model(
    features_path: str = FEATURES_CSV,
    model_save_path: str = MODEL_SAVE_PATH,
    contamination: float = 0.01,
    random_state: int = 42,
) -> None:
    """
    Train a one-class anomaly detection model on real voice features only.
    Uses Isolation Forest: inliers (real) = 1, outliers (fake) = -1.
    Saves the trained model and fitted scaler to model_save_path.

    Args:
        features_path: Path to features/features.csv.
        model_save_path: Path to save the pickle (model + scaler).
        contamination: Expected proportion of outliers (float in (0, 0.5)); keep low for real-only data.
        random_state: For reproducibility.
    """
    path = Path(features_path)
    if not path.is_file():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_csv(path)
    feature_cols = _get_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in the features CSV.")

    X = df[feature_cols].values
    n_samples = len(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples="auto",
    )
    model.fit(X_scaled)

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_cols,
    }
    with open(save_path, "wb") as f:
        pickle.dump(artifact, f)

    print("Training completed successfully.")
    print(f"Number of samples used: {n_samples}")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else FEATURES_CSV
    mp = sys.argv[2] if len(sys.argv) > 2 else MODEL_SAVE_PATH
    train_model(features_path=fp, model_save_path=mp)
