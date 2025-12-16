"""
Inference module for music genre classification.
Handles predictions from both baseline and CNN models.
"""
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODELS_DIR, GENRES, IDX_TO_GENRE, GENRE_TO_IDX,
    SAMPLE_RATE, SEGMENT_DURATION
)
from src.audio_features import (
    load_audio, segment_audio, extract_mel_spectrogram,
    extract_baseline_features, normalize_spectrogram
)
from src.model import load_model, GenreCNN


class GenrePredictor:
    """
    Unified predictor class for genre classification.
    Supports both baseline models and CNN.
    """
    
    def __init__(
        self,
        model_type: str = "cnn",
        model_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_type: Type of model ("cnn", "logistic_regression", "knn", "random_forest")
            model_path: Path to model file (uses default if None)
            device: Device for CNN inference
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        
        # Load model
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str]):
        """Load the specified model."""
        if model_path:
            path = Path(model_path)
        else:
            # Use default paths
            model_files = {
                "cnn": MODELS_DIR / "cnn_model.pth",
                "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
                "knn": MODELS_DIR / "knn.pkl",
                "random_forest": MODELS_DIR / "random_forest.pkl"
            }
            path = model_files.get(self.model_type)
        
        if path is None or not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        if self.model_type == "cnn":
            self.model = load_model(str(path), self.device)
        else:
            self.model = joblib.load(path)
            # Load scaler for baseline models
            scaler_path = MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
    
    def predict(
        self,
        audio_path: str,
        return_probabilities: bool = True,
        aggregate: str = "vote"
    ) -> Dict:
        """
        Predict genre for an audio file.
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Whether to return class probabilities
            aggregate: How to aggregate segment predictions ("vote", "mean")
        
        Returns:
            Dictionary with prediction results
        """
        if self.model_type == "cnn":
            return self._predict_cnn(audio_path, return_probabilities, aggregate)
        else:
            return self._predict_baseline(audio_path, return_probabilities)
    
    def _predict_cnn(
        self,
        audio_path: str,
        return_probabilities: bool,
        aggregate: str
    ) -> Dict:
        """CNN prediction pipeline."""
        # Load and segment audio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = load_audio(audio_path)
        
        segments = segment_audio(y, sr, SEGMENT_DURATION)
        
        if not segments:
            raise ValueError("Audio file too short for prediction")
        
        # Extract spectrograms
        spectrograms = []
        for segment in segments:
            mel_spec = extract_mel_spectrogram(segment, sr)
            mel_spec_norm = normalize_spectrogram(mel_spec)
            spectrograms.append(mel_spec_norm)
        
        # Convert to tensor
        specs_tensor = torch.FloatTensor(np.array(spectrograms)).unsqueeze(1)
        specs_tensor = specs_tensor.to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(specs_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Aggregate predictions
        if aggregate == "vote":
            # Majority voting
            unique, counts = np.unique(predictions, return_counts=True)
            final_pred = unique[np.argmax(counts)]
            confidence = counts.max() / len(predictions)
        else:
            # Mean probability
            mean_probs = probabilities.mean(axis=0)
            final_pred = np.argmax(mean_probs)
            confidence = mean_probs[final_pred]
        
        result = {
            "predicted_genre": IDX_TO_GENRE[final_pred],
            "predicted_index": int(final_pred),
            "confidence": float(confidence),
            "num_segments": len(segments),
            "segment_predictions": [IDX_TO_GENRE[p] for p in predictions]
        }
        
        if return_probabilities:
            mean_probs = probabilities.mean(axis=0)
            result["probabilities"] = {
                genre: float(mean_probs[idx])
                for genre, idx in GENRE_TO_IDX.items()
            }
            result["top_3"] = self._get_top_k(mean_probs, k=3)
        
        return result
    
    def _predict_baseline(
        self,
        audio_path: str,
        return_probabilities: bool
    ) -> Dict:
        """Baseline model prediction pipeline."""
        # Load audio and extract features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = load_audio(audio_path)
        
        features = extract_baseline_features(y, sr)
        
        # Convert to array
        feature_names = sorted(features.keys())
        X = np.array([[features[name] for name in feature_names]])
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        
        result = {
            "predicted_genre": IDX_TO_GENRE[prediction],
            "predicted_index": int(prediction),
            "model_type": self.model_type
        }
        
        # Get probabilities if available
        if return_probabilities and hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            result["confidence"] = float(probabilities[prediction])
            result["probabilities"] = {
                genre: float(probabilities[idx])
                for genre, idx in GENRE_TO_IDX.items()
            }
            result["top_3"] = self._get_top_k(probabilities, k=3)
        
        return result
    
    def _get_top_k(self, probabilities: np.ndarray, k: int = 3) -> List[Dict]:
        """Get top-k predictions with probabilities."""
        top_indices = np.argsort(probabilities)[::-1][:k]
        return [
            {
                "genre": IDX_TO_GENRE[idx],
                "probability": float(probabilities[idx])
            }
            for idx in top_indices
        ]
    
    def predict_batch(
        self,
        audio_paths: List[str],
        return_probabilities: bool = True
    ) -> List[Dict]:
        """Predict genres for multiple audio files."""
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path, return_probabilities)
                result["file"] = path
                result["status"] = "success"
            except Exception as e:
                result = {
                    "file": path,
                    "status": "error",
                    "error": str(e)
                }
            results.append(result)
        return results


def quick_predict(audio_path: str, model_type: str = "cnn") -> str:
    """
    Quick prediction function.
    
    Args:
        audio_path: Path to audio file
        model_type: Model to use for prediction
    
    Returns:
        Predicted genre name
    """
    predictor = GenrePredictor(model_type=model_type)
    result = predictor.predict(audio_path, return_probabilities=False)
    return result["predicted_genre"]


def get_available_models() -> List[str]:
    """Get list of available trained models."""
    available = []
    
    model_files = {
        "cnn": MODELS_DIR / "cnn_model.pth",
        "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
        "knn": MODELS_DIR / "knn.pkl",
        "random_forest": MODELS_DIR / "random_forest.pkl"
    }
    
    for name, path in model_files.items():
        if path.exists():
            available.append(name)
    
    return available


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict music genre")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--model", "-m",
        choices=["cnn", "logistic_regression", "knn", "random_forest"],
        default="cnn",
        help="Model to use for prediction"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    # Check available models
    available = get_available_models()
    if not available:
        print("No trained models found. Please run training first.")
        exit(1)
    
    if args.model not in available:
        print(f"Model '{args.model}' not found. Available models: {available}")
        exit(1)
    
    # Make prediction
    predictor = GenrePredictor(model_type=args.model)
    result = predictor.predict(args.audio_file)
    
    print(f"\nPredicted Genre: {result['predicted_genre'].upper()}")
    print(f"Confidence: {result.get('confidence', 'N/A'):.2%}")
    
    if args.verbose and "top_3" in result:
        print("\nTop 3 predictions:")
        for i, pred in enumerate(result["top_3"], 1):
            print(f"  {i}. {pred['genre']}: {pred['probability']:.2%}")
