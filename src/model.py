"""
CNN model architecture for music genre classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CNN_CONFIG, NUM_GENRES


class GenreCNN(nn.Module):
    """
    Convolutional Neural Network for music genre classification.
    
    Architecture:
        - 3 convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
        - Global Average Pooling
        - Dense layer with dropout
        - Output softmax layer
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        n_mels: int = 128,
        conv_filters: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        pool_size: int = 2,
        dense_units: int = 128,
        dropout_rate: float = 0.5,
        num_classes: int = NUM_GENRES
    ):
        super(GenreCNN, self).__init__()
        
        self.conv_filters = conv_filters
        self.num_classes = num_classes
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_filters:
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_filters[-1], dense_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_units, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, n_mels, time_frames)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_model_state)
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


def create_model(config: dict = None) -> GenreCNN:
    """
    Factory function to create model from config.
    
    Args:
        config: Model configuration dict (uses CNN_CONFIG if None)
    
    Returns:
        Initialized GenreCNN model
    """
    if config is None:
        config = CNN_CONFIG
    
    model = GenreCNN(
        input_channels=config.get("input_shape", (1, 128, 130))[0],
        n_mels=config.get("input_shape", (1, 128, 130))[1],
        conv_filters=tuple(config.get("conv_filters", [32, 64, 128])),
        kernel_size=config.get("kernel_size", 3),
        pool_size=config.get("pool_size", 2),
        dense_units=config.get("dense_units", 128),
        dropout_rate=config.get("dropout_rate", 0.5),
        num_classes=config.get("num_classes", NUM_GENRES)
    )
    
    return model


def load_model(model_path: str, device: str = "cpu") -> GenreCNN:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model weights
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def save_model(model: GenreCNN, model_path: str):
    """Save model weights to disk."""
    torch.save(model.state_dict(), model_path)
