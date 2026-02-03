"""
VoiceShield - Audio preprocessing module.
Converts audio to mono, 16 kHz, normalized amplitude; returns clean NumPy waveform.
"""

import numpy as np
import librosa


TARGET_SR = 16000


def load_and_preprocess(audio_path: str) -> np.ndarray:
    """
    Load audio file, convert to mono, resample to 16 kHz, normalize amplitude.
    Returns a clean NumPy waveform suitable for feature extraction.

    Args:
        audio_path: Path to the audio file (e.g. .wav, .mp3).

    Returns:
        One-dimensional NumPy array (float) of the preprocessed waveform.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        librosa.util.exceptions.ParameterError: If file is not valid audio.
    """
    # Load: mono=True, resample to TARGET_SR
    waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    # Normalize amplitude (peak normalization to [-1, 1] range)
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform.astype(np.float64)


def preprocess_from_array(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Preprocess an already-loaded waveform: ensure mono, 16 kHz, normalized.
    Useful when audio is loaded elsewhere (e.g. API) and passed as array.

    Args:
        waveform: 1D or 2D array (samples, channels).
        sr: Original sample rate.

    Returns:
        Preprocessed 1D NumPy array at 16 kHz, normalized.
    """
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != TARGET_SR:
        waveform = librosa.resample(
            waveform.astype(np.float64), orig_sr=sr, target_sr=TARGET_SR
        )
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform.astype(np.float64)
