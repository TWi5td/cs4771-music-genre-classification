# 🎵 Music Genre Classification

**CS 4771 Machine Learning Project** | University of Idaho | Thomas Schmidt

A complete machine learning system for classifying music genres from audio files, featuring both traditional ML baselines and deep learning with CNN. Includes a web-based interface for easy interaction.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [API Reference](#api-reference)
- [Development](#development)

---

## Overview

This project implements multiple approaches to music genre classification:

- **Baseline Models**: Logistic Regression, k-NN, Random Forest using handcrafted audio features
- **Deep Learning**: CNN trained on mel spectrograms for end-to-end learning
- **Web Interface**: Modern, responsive UI for uploading and classifying audio files

**Supported Genres**: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

---

## Features

✅ Multiple classification models (CNN, Random Forest, Logistic Regression, k-NN)  
✅ Web-based interface with drag-and-drop file upload  
✅ REST API for programmatic access  
✅ Cross-platform support (Windows 11 & Ubuntu Server)  
✅ Real-time classification with confidence scores  
✅ Detailed probability breakdown for all genres  

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- FFmpeg (for audio processing)

### Quick Start

#### Windows 11

```bash
# Clone the repository
git clone https://github.com/TWi5td/cs4771-music-genre-classification.git
cd cs4771-music-genre-classification

# Run setup script
scripts\setup_windows.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Ubuntu Server (Headless)

```bash
# Clone the repository
git clone https://github.com/TWi5td/cs4771-music-genre-classification.git
cd cs4771-music-genre-classification

# Run setup script
chmod +x scripts/setup_server.sh
./scripts/setup_server.sh

# Or manually:
sudo apt-get install python3-venv libsndfile1 ffmpeg
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download Dataset

```bash
# Activate virtual environment first
python scripts/download_dataset.py

# Or download manually from Kaggle:
# https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
```

### Preprocess & Train

```bash
# Extract audio features
python src/preprocess.py

# Train all models
python src/train.py
```

---

## Usage

### Web Interface

Start the Flask server:

```bash
# Development mode
python app.py

# Production mode (Ubuntu)
gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
```

Open your browser to `http://localhost:5000` (or your server IP).

### Command Line

```bash
# Classify a single file
python src/predict.py path/to/audio.wav

# Use a specific model
python src/predict.py path/to/audio.wav --model random_forest

# Verbose output with top-3 predictions
python src/predict.py path/to/audio.wav -v
```

### Python API

```python
from src.predict import GenrePredictor

# Initialize predictor
predictor = GenrePredictor(model_type="cnn")

# Classify audio file
result = predictor.predict("path/to/song.wav")

print(f"Genre: {result['predicted_genre']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top 3: {result['top_3']}")
```

---

## Project Structure

```
cs4771-music-genre-classification/
├── app.py                 # Flask web application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── audio_features.py  # Audio feature extraction
│   ├── model.py           # CNN architecture
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Training scripts
│   └── predict.py         # Inference module
│
├── templates/             # HTML templates
│   └── index.html
│
├── static/                # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── scripts/               # Setup scripts
│   ├── setup_server.sh    # Ubuntu setup
│   ├── setup_windows.bat  # Windows setup
│   └── download_dataset.py
│
├── data/                  # Dataset (gitignored)
│   ├── raw/               # GTZAN audio files
│   └── processed/         # Extracted features
│
├── models/                # Trained models (gitignored)
│   ├── cnn_model.pth
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   └── scaler.pkl
│
├── notebooks/             # Jupyter notebooks
│   └── baseline_models.ipynb
│
└── tests/                 # Unit tests
```

---

## Models

### CNN (Convolutional Neural Network)

- **Input**: 128×130 log-mel spectrograms from 3-second audio segments
- **Architecture**: 3 conv blocks (32→64→128 filters) with batch norm and max pooling
- **Output**: 10-class softmax predictions
- **Training**: Adam optimizer, early stopping, dropout regularization

### Baseline Models

| Model | Features | Description |
|-------|----------|-------------|
| Random Forest | MFCCs, chroma, spectral | Ensemble of 100 decision trees |
| Logistic Regression | MFCCs, chroma, spectral | L2-regularized linear classifier |
| k-NN | MFCCs, chroma, spectral | k=5 with cosine distance |

### Audio Features Extracted

- **MFCCs** (13 coefficients): Mel-frequency cepstral coefficients
- **Chroma**: Pitch class distribution
- **Spectral**: Centroid, bandwidth, rolloff, contrast
- **Temporal**: Zero-crossing rate, RMS energy, tempo

---

## API Reference

### POST `/api/predict`

Classify an audio file.

**Request:**
- `audio`: Audio file (multipart/form-data)
- `model`: Model type (optional, default: "cnn")

**Response:**
```json
{
  "predicted_genre": "rock",
  "predicted_index": 9,
  "confidence": 0.87,
  "num_segments": 10,
  "probabilities": {
    "blues": 0.02,
    "classical": 0.01,
    "rock": 0.87,
    ...
  },
  "top_3": [
    {"genre": "rock", "probability": 0.87},
    {"genre": "metal", "probability": 0.08},
    {"genre": "blues", "probability": 0.02}
  ]
}
```

### GET `/api/models`

List available trained models.

### GET `/api/genres`

List supported genres.

### GET `/api/health`

Health check endpoint.

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ app.py
isort src/ app.py
flake8 src/ app.py
```

### Running as Systemd Service (Ubuntu)

After running `setup_server.sh`:

```bash
sudo systemctl daemon-reload
sudo systemctl start music-genre-classifier
sudo systemctl enable music-genre-classifier  # Auto-start on boot

# View logs
sudo journalctl -u music-genre-classifier -f
```

---

## References

1. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293-302.
2. McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*.
3. Choi, K., et al. (2017). Convolutional recurrent neural networks for music classification. *ICASSP 2017*.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Author**: Thomas Schmidt  
**Course**: CS 4771 - Machine Learning  
**Institution**: University of Idaho  
**Date**: Fall 2025
