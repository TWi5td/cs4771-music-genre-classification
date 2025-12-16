"""
Music Genre Classification - Source Package
CS 4771 Machine Learning Project
"""

from .audio_features import (
    load_audio,
    extract_mel_spectrogram,
    extract_mfcc,
    extract_baseline_features,
    segment_audio,
    normalize_spectrogram
)

from .model import (
    GenreCNN,
    create_model,
    load_model,
    save_model,
    EarlyStopping
)

from .predict import (
    GenrePredictor,
    quick_predict,
    get_available_models
)

__all__ = [
    # Audio features
    'load_audio',
    'extract_mel_spectrogram',
    'extract_mfcc',
    'extract_baseline_features',
    'segment_audio',
    'normalize_spectrogram',
    # Model
    'GenreCNN',
    'create_model',
    'load_model',
    'save_model',
    'EarlyStopping',
    # Prediction
    'GenrePredictor',
    'quick_predict',
    'get_available_models',
]
