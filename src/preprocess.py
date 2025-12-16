"""
Data preprocessing script for GTZAN dataset.
Extracts features and prepares data for training.
"""
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict
import warnings

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, GENRES, GENRE_TO_IDX,
    SAMPLE_RATE, SEGMENT_DURATION, N_MELS
)
from src.audio_features import (
    load_audio, segment_audio, extract_mel_spectrogram,
    extract_baseline_features, normalize_spectrogram
)


def find_audio_files(data_dir: Path) -> List[Tuple[Path, str]]:
    """
    Find all audio files in the dataset directory.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        List of (file_path, genre) tuples
    """
    audio_files = []
    
    # GTZAN structure: genres/genre_name/genre.XXXXX.wav
    for genre in GENRES:
        genre_dir = data_dir / genre
        if not genre_dir.exists():
            # Try alternative structure: genres/genre_name.XXXXX.wav
            genre_dir = data_dir / "genres" / genre
        
        if genre_dir.exists():
            for audio_file in genre_dir.glob("*.wav"):
                audio_files.append((audio_file, genre))
        else:
            # Check if files are in root with genre prefix
            for audio_file in data_dir.glob(f"{genre}.*.wav"):
                audio_files.append((audio_file, genre))
    
    # Also check for flat structure
    if not audio_files:
        for genre in GENRES:
            for audio_file in data_dir.glob(f"**/{genre}*.wav"):
                audio_files.append((audio_file, genre))
    
    return audio_files


def preprocess_for_baseline(
    audio_files: List[Tuple[Path, str]],
    output_path: Path
) -> pd.DataFrame:
    """
    Extract baseline features from all audio files.
    
    Args:
        audio_files: List of (file_path, genre) tuples
        output_path: Path to save CSV
    
    Returns:
        DataFrame with extracted features
    """
    all_features = []
    
    print("Extracting baseline features...")
    for file_path, genre in tqdm(audio_files):
        try:
            features = extract_baseline_features_safe(file_path)
            if features:
                features["genre"] = genre
                features["filename"] = file_path.name
                all_features.append(features)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    df = pd.DataFrame(all_features)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
    
    return df


def extract_baseline_features_safe(file_path: Path) -> Dict:
    """Extract features with error handling."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = load_audio(str(file_path))
            if len(y) < sr:  # Less than 1 second
                return None
            features = extract_baseline_features(y, sr)
            return features
    except Exception:
        return None


def preprocess_for_cnn(
    audio_files: List[Tuple[Path, str]],
    output_path: Path,
    segment_duration: float = SEGMENT_DURATION
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mel spectrograms from all audio files.
    
    Args:
        audio_files: List of (file_path, genre) tuples
        output_path: Path to save HDF5 file
        segment_duration: Duration of each segment in seconds
    
    Returns:
        Tuple of (spectrograms, labels)
    """
    all_specs = []
    all_labels = []
    
    print("Extracting mel spectrograms...")
    for file_path, genre in tqdm(audio_files):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = load_audio(str(file_path))
            
            segments = segment_audio(y, sr, segment_duration)
            label = GENRE_TO_IDX[genre]
            
            for segment in segments:
                mel_spec = extract_mel_spectrogram(segment, sr)
                mel_spec_norm = normalize_spectrogram(mel_spec)
                all_specs.append(mel_spec_norm)
                all_labels.append(label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Convert to arrays
    spectrograms = np.array(all_specs, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    
    # Save to HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("spectrograms", data=spectrograms, compression="gzip")
        f.create_dataset("labels", data=labels)
    
    print(f"Saved {len(labels)} spectrograms to {output_path}")
    print(f"Spectrogram shape: {spectrograms.shape}")
    
    return spectrograms, labels


def download_gtzan_dataset(output_dir: Path) -> bool:
    """
    Download GTZAN dataset (placeholder - actual download requires manual steps).
    
    The GTZAN dataset can be downloaded from:
    - http://marsyas.info/downloads/datasets.html
    - https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
    
    Returns:
        True if download successful
    """
    print("=" * 60)
    print("GTZAN Dataset Download Instructions")
    print("=" * 60)
    print("""
The GTZAN dataset needs to be downloaded manually:

Option 1: Kaggle (Recommended)
  1. Visit: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
  2. Download and extract to: {data_dir}
  3. Ensure structure is: {data_dir}/genres/<genre_name>/<files>.wav

Option 2: Direct Download
  1. Visit: http://marsyas.info/downloads/datasets.html
  2. Download genres.tar.gz
  3. Extract to: {data_dir}

Expected structure:
{data_dir}/
├── genres/
│   ├── blues/
│   │   ├── blues.00000.wav
│   │   └── ...
│   ├── classical/
│   └── ...
""".format(data_dir=output_dir))
    print("=" * 60)
    return False


def verify_dataset(data_dir: Path) -> Dict:
    """
    Verify dataset integrity and return statistics.
    
    Returns:
        Dictionary with dataset statistics
    """
    audio_files = find_audio_files(data_dir)
    
    stats = {
        "total_files": len(audio_files),
        "genres": {},
        "valid": len(audio_files) > 0
    }
    
    for _, genre in audio_files:
        stats["genres"][genre] = stats["genres"].get(genre, 0) + 1
    
    return stats


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("Music Genre Classification - Data Preprocessing")
    print("=" * 60)
    
    # Check for existing data
    audio_files = find_audio_files(RAW_DATA_DIR)
    
    if not audio_files:
        print(f"\nNo audio files found in {RAW_DATA_DIR}")
        download_gtzan_dataset(RAW_DATA_DIR)
        return
    
    # Verify dataset
    stats = verify_dataset(RAW_DATA_DIR)
    print(f"\nFound {stats['total_files']} audio files")
    print("Genre distribution:")
    for genre, count in sorted(stats["genres"].items()):
        print(f"  {genre}: {count}")
    
    # Create output directory
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract baseline features
    baseline_path = PROCESSED_DATA_DIR / "baseline_features.csv"
    preprocess_for_baseline(audio_files, baseline_path)
    
    # Extract spectrograms for CNN
    specs_path = PROCESSED_DATA_DIR / "spectrograms.h5"
    preprocess_for_cnn(audio_files, specs_path)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Baseline features: {baseline_path}")
    print(f"Spectrograms: {specs_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
