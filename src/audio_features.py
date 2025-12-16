"""
Audio feature extraction utilities for music genre classification.
Handles loading, preprocessing, and feature extraction from audio files.
"""
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SAMPLE_RATE, SEGMENT_DURATION, HOP_LENGTH,
    N_MELS, N_MFCC, N_FFT
)


def load_audio(
    file_path: str,
    sr: int = SAMPLE_RATE,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return the waveform and sample rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        duration: Duration to load (None for full file)
        offset: Start reading after this time (seconds)
    
    Returns:
        Tuple of (audio_waveform, sample_rate)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr_loaded = librosa.load(
            file_path,
            sr=sr,
            duration=duration,
            offset=offset,
            mono=True
        )
    return y, sr_loaded


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Extract log-mel spectrogram from audio waveform.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
    
    Returns:
        Log-mel spectrogram (n_mels x time_frames)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def extract_mfcc(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Extract MFCC features from audio waveform.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCCs to extract
        n_fft: FFT window size
        hop_length: Hop length between frames
    
    Returns:
        MFCC features (n_mfcc x time_frames)
    """
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfccs


def extract_baseline_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """
    Extract statistical features for baseline models.
    
    Args:
        y: Audio waveform
        sr: Sample rate
    
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i}_std"] = np.std(mfccs[i])
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_centroid_std"] = np.std(spectral_centroid)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
    features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
    features["spectral_rolloff_std"] = np.std(spectral_rolloff)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast_mean"] = np.mean(spectral_contrast)
    features["spectral_contrast_std"] = np.std(spectral_contrast)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)
    
    # RMS energy
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    
    return features


def segment_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    segment_duration: float = SEGMENT_DURATION
) -> List[np.ndarray]:
    """
    Split audio into non-overlapping segments.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        segment_duration: Duration of each segment in seconds
    
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sr)
    segments = []
    
    for start in range(0, len(y) - segment_samples + 1, segment_samples):
        segment = y[start:start + segment_samples]
        segments.append(segment)
    
    return segments


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram to [0, 1] range.
    
    Args:
        spec: Input spectrogram
    
    Returns:
        Normalized spectrogram
    """
    spec_min = spec.min()
    spec_max = spec.max()
    if spec_max - spec_min > 0:
        return (spec - spec_min) / (spec_max - spec_min)
    return spec - spec_min


def process_audio_for_cnn(
    file_path: str,
    segment_duration: float = SEGMENT_DURATION
) -> List[np.ndarray]:
    """
    Process audio file and return mel spectrograms for CNN input.
    
    Args:
        file_path: Path to audio file
        segment_duration: Duration of each segment
    
    Returns:
        List of normalized mel spectrograms
    """
    y, sr = load_audio(file_path)
    segments = segment_audio(y, sr, segment_duration)
    
    spectrograms = []
    for segment in segments:
        mel_spec = extract_mel_spectrogram(segment, sr)
        mel_spec_norm = normalize_spectrogram(mel_spec)
        spectrograms.append(mel_spec_norm)
    
    return spectrograms


def process_audio_for_baseline(file_path: str) -> Dict[str, float]:
    """
    Process audio file and return features for baseline models.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Dictionary of aggregated features
    """
    y, sr = load_audio(file_path)
    features = extract_baseline_features(y, sr)
    return features


# Data augmentation functions
def time_stretch(y: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Apply time stretching to audio."""
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Apply pitch shifting to audio."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add random noise to audio."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise
