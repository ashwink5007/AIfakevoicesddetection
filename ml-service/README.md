# VoiceShield – AI-Generated Voice Detection

VoiceShield is an end-to-end machine learning pipeline that **detects AI-generated (deepfake) voices** using **only real human voice samples** for training. It treats the problem as **one-class anomaly detection**: the model learns the distribution of genuine human speech; any voice that deviates from these patterns is classified as **FAKE**.

---

## How the Model Works

1. **Training data**: Only **real** human voice recordings. No fake or synthetic voices are used.
2. **Learning objective**: The model learns what “normal” (real) human speech sounds like in terms of acoustic features (MFCCs, pitch, jitter, energy).
3. **At prediction time**: A new voice is compared to this learned distribution. If it fits the distribution of real voices → **REAL**. If it deviates (anomaly) → **FAKE**.

**Classification rule**: If a voice does **not** resemble the real human voice patterns learned during training, it is classified as **FAKE**.

---

## Why One-Class Learning?

- **No fake data at training time**: We do not have (or do not trust) labeled deepfake samples.
- **Anomaly detection**: “Real” is the only class we can define; everything else is treated as an anomaly (potential fake).
- **Algorithm**: The pipeline uses **Isolation Forest**, which learns the density of real voices and flags outliers. Other options (e.g. One-Class SVM) can be swapped in for smaller datasets.

---

## Setup

### 1. Python

Use **Python 3.10** (or compatible 3.x).

### 2. Install Dependencies

From the `ml-service` directory:

```bash
pip install -r requirements.txt
```

All required libraries (numpy, pandas, librosa, parselmouth, scikit-learn, joblib, soundfile) will be installed. The project is designed to work with these dependencies only.

---

## How to Train the Model

Run these steps **from the `ml-service` directory** so that paths like `features/features.csv` and `models/voice_model.pkl` resolve correctly.

### Step 1: Extract features from real voice data

Place your **real human voice** recordings in a folder (e.g. `data/`). Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`.

```bash
python src/extract_features.py data
```

- Reads all supported audio files under `data/`.
- Preprocesses (mono, 16 kHz, normalized) and extracts: **MFCCs (13)**, **pitch**, **jitter**, **energy (RMS)**.
- Writes one row per file to **`features/features.csv`**.

Optional: custom output CSV and/or limit number of files (e.g. for quick testing on large datasets):

```bash
python src/extract_features.py data features/my_features.csv
python src/extract_features.py data features/features.csv 500
```

### Step 2: Train the one-class model

```bash
python src/train_model.py
```

- Loads **`features/features.csv`**.
- Trains an **Isolation Forest** (one-class) on these real-voice features.
- Saves the trained model and scaler to **`models/voice_model.pkl`**.
- Prints a short summary (e.g. “Training completed”, number of samples).

Custom paths:

```bash
python src/train_model.py features/features.csv models/voice_model.pkl
```

---

## How to Run Prediction

After training, you can classify a single audio file as **REAL** or **FAKE**:

```bash
python src/predict.py path/to/audio.wav
```

Example output:

```
Prediction: REAL
Confidence: 91%
```

- The script preprocesses the audio, extracts the same features, loads **`models/voice_model.pkl`**, and runs the one-class model.
- **REAL** = inlier (matches learned real-voice distribution).
- **FAKE** = outlier (anomaly; does not match real-voice patterns).
- **Confidence** is derived from the model’s decision function and expressed as a percentage.

Optional: specify a custom model path:

```bash
python src/predict.py path/to/audio.wav models/voice_model.pkl
```

---

## Web Application

Run the model as a web app to test it in the browser:

```bash
# From ml-service (after training the model once)
python run_web.py
# or: uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open in your browser:

- **http://localhost:8000** – upload form and run REAL/FAKE check
- **http://127.0.0.1:8000** – same (local)
- **http://0.0.0.0:8000** – from another device on your network, use your machine’s IP (e.g. http://192.168.1.10:8000)

The page lets you upload an audio file and shows **Prediction: REAL** or **FAKE** with **Confidence: X%**. The API also exposes:
- `GET /` – upload form
- `POST /predict` – multipart file upload, returns `{ "prediction": "REAL"|"FAKE", "confidence": 91 }`
- `GET /health` – health check and whether the model is loaded

---

## Example End-to-End Commands

From **`ml-service`**:

```bash
# 1. Install
pip install -r requirements.txt

# 2. Extract features (real voices in data/)
python src/extract_features.py data

# 3. Train and save model
python src/train_model.py

# 4. Predict on a file
python src/predict.py path/to/test_voice.wav

# 5. Run web app (optional)
python run_web.py
```

---

## Project Layout

```
ml-service/
├── requirements.txt       # Python dependencies (Python 3.10 compatible)
├── README.md              # This file
├── data/                  # Put real voice recordings here (or point extract_features to another dir)
├── features/
│   └── features.csv       # Extracted features (created by extract_features.py)
├── models/
│   └── voice_model.pkl    # Trained model + scaler (created by train_model.py)
└── src/
    ├── preprocess.py      # Mono, 16 kHz, normalize → NumPy waveform
    ├── extract_features.py # MFCCs, pitch, jitter, energy → features.csv
    ├── train_model.py     # One-class training → voice_model.pkl
    └── predict.py        # Load model, predict REAL/FAKE + confidence
```

---

## Integration (FastAPI / MERN)

The pipeline is modular and script-friendly:

- **Preprocessing**: `preprocess.load_and_preprocess(audio_path)` → waveform.
- **Features**: `extract_features.extract_features_from_waveform(waveform)` or `extract_features_from_file(audio_path)`.
- **Training**: `train_model.train_model(features_path, model_save_path)`.
- **Prediction**: `predict.predict(audio_path, model_path)` → `(label, confidence)`.

You can call these from a FastAPI backend (e.g. upload file → save → `predict.predict(path)`) or from any other service. The same model and scaler in `voice_model.pkl` ensure consistent behavior.

---

## Constraints and Design

- **No deepfake data in training**: Only real human voices are used to fit the model.
- **One-class rule**: Anything that does not resemble the learned real-voice distribution is classified as **FAKE**.
- Code is modular and reusable for future web/API integration.
