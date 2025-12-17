#!/usr/bin/env python3
"""
GTZAN Dataset Download Helper
Downloads and extracts the GTZAN dataset for music genre classification.
"""
import os
import sys
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from config import RAW_DATA_DIR
except ImportError:
    RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# Dataset URLs (backup mirrors)
GTZAN_URLS = [
    # Primary: Marsyas (official)
    "http://opihi.cs.uvic.ca/sound/genres.tar.gz",
    # Backup mirrors (if available)
]

KAGGLE_INSTRUCTIONS = """
================================================================================
KAGGLE DOWNLOAD (RECOMMENDED)
================================================================================

The GTZAN dataset is most reliably available on Kaggle:

1. Install Kaggle CLI:
   pip install kaggle

2. Setup Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save kaggle.json to:
     - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
     - Linux/Mac: ~/.kaggle/kaggle.json

3. Download the dataset:
   kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification

4. Extract to: {data_dir}

5. Ensure structure is:
   {data_dir}/
   └── genres/
       ├── blues/
       │   ├── blues.00000.wav
       │   └── ...
       ├── classical/
       └── ...

================================================================================
"""


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    print(f"\n{desc}: {url}")
    
    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        bar = '=' * (percent // 2) + '>' + ' ' * (50 - percent // 2)
        sys.stdout.write(f'\r[{bar}] {percent}%')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest, progress_hook)
        print()  # New line after progress bar
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract tar.gz or zip archive."""
    print(f"Extracting to {dest_dir}...")
    
    if str(archive_path).endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(dest_dir)
    elif str(archive_path).endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")
    
    print("Extraction complete!")


def verify_dataset(data_dir: Path) -> bool:
    """Verify the dataset was extracted correctly."""
    genres_dir = data_dir / "genres"
    if not genres_dir.exists():
        # Check if files are directly in data_dir
        genres_dir = data_dir
    
    expected_genres = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]
    
    found_genres = []
    for genre in expected_genres:
        genre_path = genres_dir / genre
        if genre_path.exists() and any(genre_path.glob("*.wav")):
            found_genres.append(genre)
    
    if len(found_genres) == len(expected_genres):
        print(f"\n✓ Dataset verified: {len(found_genres)} genres found")
        return True
    elif found_genres:
        print(f"\n⚠ Partial dataset: {len(found_genres)}/{len(expected_genres)} genres")
        print(f"  Found: {found_genres}")
        return True
    else:
        print("\n✗ Dataset verification failed: No genres found")
        return False


def download_gtzan():
    """Main download function."""
    print("=" * 60)
    print("GTZAN Dataset Downloader")
    print("=" * 60)
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists
    if verify_dataset(RAW_DATA_DIR):
        print("\nDataset already exists. Skipping download.")
        return True
    
    # Try downloading from available URLs
    archive_path = RAW_DATA_DIR / "genres.tar.gz"
    
    for url in GTZAN_URLS:
        if download_file(url, archive_path, "Downloading GTZAN dataset"):
            try:
                extract_archive(archive_path, RAW_DATA_DIR)
                archive_path.unlink()  # Remove archive after extraction
                
                if verify_dataset(RAW_DATA_DIR):
                    return True
            except Exception as e:
                print(f"Error extracting: {e}")
    
    print("Attempting Kaggle download...")
    if download_from_kaggle(RAW_DATA_DIR):
        return verify_dataset(RAW_DATA_DIR)
    
    print(KAGGLE_INSTRUCTIONS.format(data_dir=RAW_DATA_DIR))
    return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download GTZAN dataset")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=RAW_DATA_DIR,
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--kaggle", "-k",
        action="store_true",
        help="Show Kaggle download instructions"
    )
    
    args = parser.parse_args()
    
    if args.kaggle:
        print(KAGGLE_INSTRUCTIONS.format(data_dir=args.output))
        return
    
    global RAW_DATA_DIR
    RAW_DATA_DIR = args.output
    
    success = download_gtzan()
    
    if success:
        print("\n" + "=" * 60)
        print("Dataset ready! Next steps:")
        print("  1. python src/preprocess.py")
        print("  2. python src/train.py")
        print("  3. python app.py")
        print("=" * 60)
    else:
        print("\nAutomatic download failed. Please download manually.")
        sys.exit(1)

def download_from_kaggle(dest_dir: Path):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            "andradaolteanu/gtzan-dataset-music-genre-classification",
            path=dest_dir,
            unzip=True
        )
        return True
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False

if __name__ == "__main__":
    main()
