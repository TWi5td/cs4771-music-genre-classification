"""
Tests for Music Genre Classification
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GENRES, GENRE_TO_IDX, IDX_TO_GENRE, SAMPLE_RATE


class TestConfig:
    """Test configuration values."""
    
    def test_genres_count(self):
        """Test that we have 10 genres."""
        assert len(GENRES) == 10
    
    def test_genre_mapping(self):
        """Test genre to index mapping."""
        for genre in GENRES:
            idx = GENRE_TO_IDX[genre]
            assert IDX_TO_GENRE[idx] == genre
    
    def test_sample_rate(self):
        """Test sample rate is standard."""
        assert SAMPLE_RATE == 22050


class TestAudioFeatures:
    """Test audio feature extraction."""
    
    def test_normalize_spectrogram(self):
        """Test spectrogram normalization."""
        from src.audio_features import normalize_spectrogram
        
        # Create test spectrogram
        spec = np.random.randn(128, 130) * 10 + 5
        normalized = normalize_spectrogram(spec)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_segment_audio(self):
        """Test audio segmentation."""
        from src.audio_features import segment_audio
        
        # Create 30 seconds of audio at 22050 Hz
        y = np.random.randn(30 * 22050)
        segments = segment_audio(y, 22050, 3)
        
        # Should have 10 segments of 3 seconds each
        assert len(segments) == 10
        assert all(len(s) == 3 * 22050 for s in segments)


class TestModel:
    """Test model architecture."""
    
    def test_model_creation(self):
        """Test CNN model can be created."""
        from src.model import create_model
        
        model = create_model()
        assert model is not None
    
    def test_model_forward(self):
        """Test forward pass through model."""
        import torch
        from src.model import create_model
        
        model = create_model()
        model.eval()
        
        # Create batch of spectrograms (batch=2, channels=1, mels=128, time=130)
        x = torch.randn(2, 1, 128, 130)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 10)  # 2 samples, 10 genres


class TestPredictor:
    """Test predictor functionality."""
    
    def test_available_models_function(self):
        """Test get_available_models returns a list."""
        from src.predict import get_available_models
        
        models = get_available_models()
        assert isinstance(models, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
