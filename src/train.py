"""
Training script for music genre classification models.
Supports both baseline models and CNN.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json
import h5py

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODELS_DIR, PROCESSED_DATA_DIR, TRAIN_CONFIG,
    GENRES, GENRE_TO_IDX, IDX_TO_GENRE, N_MELS
)
from src.model import GenreCNN, EarlyStopping, create_model, save_model


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for mel spectrograms."""
    
    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray):
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Add channel dimension if needed
        spec = self.spectrograms[idx]
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        return spec, self.labels[idx]


class Trainer:
    """Trainer class for CNN model."""
    
    def __init__(
        self,
        model: GenreCNN,
        device: str = None,
        learning_rate: float = 0.001,
        patience: int = 10
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.early_stopping = EarlyStopping(patience=patience)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for spectrograms, labels in dataloader:
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(spectrograms)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * spectrograms.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for spectrograms, labels in dataloader:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * spectrograms.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50
    ) -> Dict:
        """Full training loop."""
        print(f"Training on {self.device}")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model and return metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for spectrograms, labels in dataloader:
                spectrograms = spectrograms.to(self.device)
                outputs = self.model(spectrograms)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "classification_report": classification_report(
                all_labels, all_preds,
                target_names=GENRES,
                output_dict=True
            ),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist()
        }


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, Dict]:
    """
    Train baseline models (Logistic Regression, k-NN, Random Forest).
    
    Returns:
        Dictionary of model results
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    
    results = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_val_scaled)
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.pkl")
    
    results["logistic_regression"] = {
        "accuracy": accuracy_score(y_val, lr_preds),
        "classification_report": classification_report(
            y_val, lr_preds, target_names=GENRES, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_val, lr_preds).tolist()
    }
    print(f"  Accuracy: {results['logistic_regression']['accuracy']:.4f}")
    
    # k-Nearest Neighbors
    print("Training k-NN...")
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_model.fit(X_train_scaled, y_train)
    knn_preds = knn_model.predict(X_val_scaled)
    joblib.dump(knn_model, MODELS_DIR / "knn.pkl")
    
    results["knn"] = {
        "accuracy": accuracy_score(y_val, knn_preds),
        "classification_report": classification_report(
            y_val, knn_preds, target_names=GENRES, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_val, knn_preds).tolist()
    }
    print(f"  Accuracy: {results['knn']['accuracy']:.4f}")
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_val_scaled)
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    
    results["random_forest"] = {
        "accuracy": accuracy_score(y_val, rf_preds),
        "classification_report": classification_report(
            y_val, rf_preds, target_names=GENRES, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_val, rf_preds).tolist()
    }
    print(f"  Accuracy: {results['random_forest']['accuracy']:.4f}")
    
    return results


def train_cnn(
    train_specs: np.ndarray,
    train_labels: np.ndarray,
    val_specs: np.ndarray,
    val_labels: np.ndarray,
    test_specs: np.ndarray = None,
    test_labels: np.ndarray = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 10
) -> Dict:
    """
    Train CNN model on spectrograms.
    
    Returns:
        Training results and metrics
    """
    # Create datasets
    train_dataset = SpectrogramDataset(train_specs, train_labels)
    val_dataset = SpectrogramDataset(val_specs, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model and trainer
    model = create_model()
    trainer = Trainer(model, learning_rate=learning_rate, patience=patience)
    
    # Train
    print("\nTraining CNN...")
    history = trainer.train(train_loader, val_loader, epochs)
    
    # Save model
    save_model(model, MODELS_DIR / "cnn_model.pth")
    
    # Evaluate
    results = {
        "history": history,
        "val_metrics": trainer.evaluate(val_loader)
    }
    
    # Test evaluation if provided
    if test_specs is not None and test_labels is not None:
        test_dataset = SpectrogramDataset(test_specs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        results["test_metrics"] = trainer.evaluate(test_loader)
        print(f"\nTest Accuracy: {results['test_metrics']['accuracy']:.4f}")
    
    print(f"Validation Accuracy: {results['val_metrics']['accuracy']:.4f}")
    
    return results


def load_processed_data() -> Tuple[Dict, Dict]:
    """
    Load preprocessed data from disk.
    
    Returns:
        Tuple of (baseline_data, cnn_data)
    """
    baseline_data = {}
    cnn_data = {}
    
    # Load baseline features
    baseline_path = PROCESSED_DATA_DIR / "baseline_features.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        baseline_data["features"] = df.drop(columns=["genre", "filename"]).values
        baseline_data["labels"] = df["genre"].map(GENRE_TO_IDX).values
    
    # Load spectrograms
    specs_path = PROCESSED_DATA_DIR / "spectrograms.h5"
    if specs_path.exists():
        with h5py.File(specs_path, "r") as f:
            cnn_data["spectrograms"] = f["spectrograms"][:]
            cnn_data["labels"] = f["labels"][:]
    
    return baseline_data, cnn_data


if __name__ == "__main__":
    # Example usage
    print("Loading processed data...")
    baseline_data, cnn_data = load_processed_data()
    
    if baseline_data:
        print("\n=== Training Baseline Models ===")
        X = baseline_data["features"]
        y = baseline_data["labels"]
        
        # Simple train/val split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        baseline_results = train_baseline_models(X_train, y_train, X_val, y_val)
        
        # Save results
        with open(MODELS_DIR / "baseline_results.json", "w") as f:
            json.dump(baseline_results, f, indent=2)
    
    if cnn_data:
        print("\n=== Training CNN Model ===")
        specs = cnn_data["spectrograms"]
        labels = cnn_data["labels"]
        
        # Train/val/test split
        from sklearn.model_selection import train_test_split
        specs_train, specs_temp, y_train, y_temp = train_test_split(
            specs, labels, test_size=0.3, stratify=labels, random_state=42
        )
        specs_val, specs_test, y_val, y_test = train_test_split(
            specs_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        cnn_results = train_cnn(
            specs_train, y_train,
            specs_val, y_val,
            specs_test, y_test,
            epochs=TRAIN_CONFIG["epochs"],
            batch_size=TRAIN_CONFIG["batch_size"],
            learning_rate=TRAIN_CONFIG["learning_rate"],
            patience=TRAIN_CONFIG["early_stopping_patience"]
        )
        
        # Save results
        with open(MODELS_DIR / "cnn_results.json", "w") as f:
            json.dump({
                "val_accuracy": cnn_results["val_metrics"]["accuracy"],
                "test_accuracy": cnn_results.get("test_metrics", {}).get("accuracy"),
                "classification_report": cnn_results["val_metrics"]["classification_report"]
            }, f, indent=2)
    
    print("\nTraining complete! Models saved to:", MODELS_DIR)
