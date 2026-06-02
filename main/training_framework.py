"""
Training framework for saliency prediction experiments.
Provides a reusable, configurable training pipeline to reduce code duplication
across different model architectures and experimental configurations.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration class for training experiments.
    Centralizes all hyperparameters and settings for easy experimentation.
    """
    experiment_name: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    model_class: str = "BiologicallyOptimizedSaliencyNet"
    loss_class: str = "BiologicallyAlignedSaliencyLoss"
    loss_kwargs: Dict[str, Any] = None
    optimizer_class: str = "Adam"
    optimizer_kwargs: Dict[str, Any] = None
    device: str = "auto"
    resume_training: bool = True
    save_best_only: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        if self.loss_kwargs is None:
            self.loss_kwargs = {}
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        if self.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging."""
        config_dict = asdict(self)
        config_dict['device'] = str(self.device)
        return config_dict


# --- BASELINE PURE KL-DIVERGENCE LOSS ---
class BiologicallyAlignedSaliencyLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(BiologicallyAlignedSaliencyLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_norm = pred_flat / (torch.sum(pred_flat, dim=1, keepdim=True) + self.eps)
        target_norm = target_flat / (torch.sum(target_flat, dim=1, keepdim=True) + self.eps)
        
        kl_divergence = target_norm * torch.log(target_norm / (pred_norm + self.eps) + self.eps)
        return torch.mean(torch.sum(kl_divergence, dim=1))


# --- REGULARIZED HYBRID LOSS FOR BACKGROUND SUPPRESSION ---
class HybridSaliencyLoss(nn.Module):
    def __init__(self, eps=1e-7, bce_weight=5.0, kl_weight=1.0):
        super(HybridSaliencyLoss, self).__init__()
        self.eps = eps
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.kl_weight = kl_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_norm = pred_flat / (torch.sum(pred_flat, dim=1, keepdim=True) + self.eps)
        target_norm = target_flat / (torch.sum(target_flat, dim=1, keepdim=True) + self.eps)
        
        kl_divergence = target_norm * torch.log(target_norm / (pred_norm + self.eps) + self.eps)
        kl_loss = torch.mean(torch.sum(kl_divergence, dim=1))

        return (self.bce_weight * bce_loss) + (self.kl_weight * kl_loss)


class TrainingFramework:
    """
    Unified training framework for saliency prediction models.
    Handles data loading, loops, checkpointing, and metrics saving.
    """
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        criterion: nn.Module,
        train_dataset,
        val_dataset,
        base_dir: str = ".",
    ):
        self.config = config
        self.model = model.to(config.device)
        self.criterion = criterion
        self.device = config.device
        self.base_dir = Path(base_dir)
        
        self._setup_paths()
        
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        self.optimizer = self._create_optimizer()
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history: List[float] = []
        self.val_history: List[float] = []
        
        if config.resume_training:
            self._load_checkpoint()
    
    def _setup_paths(self) -> None:
        self.checkpoint_path = self.base_dir / f"checkpoint_{self.config.experiment_name}.pth"
        self.best_model_path = self.base_dir / f"best_model_{self.config.experiment_name}.pth"
        self.metrics_path = self.base_dir / f"metrics_{self.config.experiment_name}.json"
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        optimizer_class = getattr(torch.optim, self.config.optimizer_class)
        optimizer_kwargs = {
            'lr': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            **self.config.optimizer_kwargs
        }
        return optimizer_class(self.model.parameters(), **optimizer_kwargs)
    
    def _load_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            if self.config.verbose:
                print(f"Loading checkpoint from: {self.checkpoint_path}")
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.train_history = checkpoint.get('train_history', [])
            self.val_history = checkpoint.get('val_history', [])
            
            if self.config.verbose:
                print(f"Resumed training from epoch {self.start_epoch}")
    
    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config.to_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
    
    def _save_best_model(self) -> None:
        torch.save(self.model.state_dict(), self.best_model_path)
        if self.config.verbose:
            print(f"Best model saved to: {self.best_model_path}")
    
    def _save_metrics(self) -> None:
        metrics = {
            'config': self.config.to_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
        }
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for images, fixations in self.train_loader:
            images, fixations = images.to(self.device), fixations.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, fixations)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        return running_loss / len(self.train_loader.dataset)
    
    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, fixations in self.val_loader:
                images, fixations = images.to(self.device), fixations.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, fixations)
                running_loss += loss.item() * images.size(0)
        return running_loss / len(self.val_loader.dataset)
    
    def train(self) -> Tuple[List[float], List[float]]:
        if self.config.verbose:
            print(f"Starting training: {self.config.experiment_name}")
            print(f"Device: {self.device}")
            print(f"Epochs: {self.start_epoch} -> {self.config.epochs}")
        
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            
            if self.config.verbose:
                print(f"Epoch {epoch + 1}/{self.config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            self._save_checkpoint(epoch)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_best_model()
                if self.config.verbose:
                    print(f"  ✓ New best validation loss: {val_loss:.4f}")
        
        if self.checkpoint_path.exists():
            os.remove(self.checkpoint_path)
        
        self._save_metrics()
        
        if self.config.verbose:
            print(f"Training completed! Best validation loss: {self.best_val_loss:.4f}")
        return self.train_history, self.val_history