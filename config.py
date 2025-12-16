"""
Configuration settings for Music Genre Classification project.
"""
import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Audio Configuration
# =============================================================================
SAMPLE_RATE = 22050  # Hz (standard for GTZAN)
SEGMENT_DURATION = 3  # seconds
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13
N_FFT = 2048

# =============================================================================
# Dataset Configuration
# =============================================================================
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]
NUM_GENRES = len(GENRES)
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}
IDX_TO_GENRE = {idx: genre for genre, idx in GENRE_TO_IDX.items()}

# =============================================================================
# Model Configuration
# =============================================================================
# CNN Architecture
CNN_CONFIG = {
    "input_shape": (1, N_MELS, 130),  # (channels, mel_bins, time_frames)
    "conv_filters": [32, 64, 128],
    "kernel_size": 3,
    "pool_size": 2,
    "dense_units": 128,
    "dropout_rate": 0.5,
    "num_classes": NUM_GENRES
}

# Training
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "early_stopping_patience": 10,
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15
}

# =============================================================================
# Web Application Configuration
# =============================================================================
FLASK_CONFIG = {
    "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-key-change-in-production"),
    "MAX_CONTENT_LENGTH": 50 * 1024 * 1024,  # 50MB max upload
    "UPLOAD_FOLDER": STATIC_DIR / "uploads",
    "ALLOWED_EXTENSIONS": {"wav", "mp3", "flac", "ogg", "m4a"}
}

# Server Configuration
SERVER_CONFIG = {
    "host": os.environ.get("HOST", "0.0.0.0"),
    "port": int(os.environ.get("PORT", 5000)),
    "debug": os.environ.get("DEBUG", "False").lower() == "true"
}
