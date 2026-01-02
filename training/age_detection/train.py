"""
Age Estimation Training Script
==============================
Train LightAgeNet on preprocessed UTKFace dataset.

Features:
- Training loop with validation
- Learning rate scheduler (ReduceLROnPlateau)
- Early stopping
- Checkpoint saving (best model)
- MAE/RMSE metrics logging
- Progress bar with live metrics
- Final test set evaluation

Usage:
    python training/age_detection/train.py
    python training/age_detection/train.py --epochs 100 --batch-size 64
    python training/age_detection/train.py --quick  # 5 epoch test run
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.age_detection.light_age_net import LightAgeNet, LightAgeNetV2, create_model
from src.contactless.age_estimation.processed_dataset import (
    ProcessedUTKFaceDataset,
    get_processed_dataloaders
)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str = "data/processed/utkface"
    batch_size: int = 32
    num_workers: int = 0  # Windows compatibility
    
    # Model
    model_type: str = "light"  # "light" or "v2"
    
    # Training
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stop_patience: int = 10
    
    # Checkpointing
    checkpoint_dir: str = "models/weights/age_detection"
    save_every: int = 10  # Save every N epochs
    
    # Logging
    log_dir: str = "logs/age_detection"


@dataclass
class TrainingMetrics:
    """Metrics tracked during training"""
    epoch: int
    train_loss: float
    train_mae: float
    val_loss: float
    val_mae: float
    val_rmse: float
    lr: float
    epoch_time: float


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[float, float]:
    """Calculate MAE and RMSE"""
    with torch.no_grad():
        mae = torch.abs(predictions - targets).mean().item()
        rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()
    return mae, rmse


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, ages in pbar:
        images = images.to(device)
        ages = ages.float().to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, ages)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        mae = torch.abs(predictions - ages).mean().item()
        total_mae += mae
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae:.2f}'})
    
    return total_loss / num_batches, total_mae / num_batches


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            ages = ages.float().to(device)
            
            predictions = model(images)
            loss = criterion(predictions, ages)
            
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(ages)
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    mae, rmse = calculate_metrics(all_predictions, all_targets)
    avg_loss = total_loss / len(loader)
    
    return avg_loss, mae, rmse


def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate on test set with detailed metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages in tqdm(loader, desc="Testing", leave=False):
            images = images.to(device)
            ages = ages.float().to(device)
            
            predictions = model(images)
            all_predictions.append(predictions)
            all_targets.append(ages)
    
    predictions = torch.cat(all_predictions).cpu().numpy()
    targets = torch.cat(all_targets).cpu().numpy()
    
    # Calculate metrics
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    metrics = {
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'std': float(np.std(abs_errors)),
        'median_error': float(np.median(abs_errors)),
        'max_error': float(np.max(abs_errors)),
        'within_5_years': float(np.mean(abs_errors <= 5) * 100),
        'within_10_years': float(np.mean(abs_errors <= 10) * 100),
    }
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: TrainingMetrics,
    path: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': asdict(metrics)
    }
    
    torch.save(checkpoint, path)
    
    if is_best:
        best_path = path.parent / 'best_model.pt'
        torch.save(model.state_dict(), best_path)
        print(f"  ✓ Saved best model (MAE: {metrics.val_mae:.2f})")


def train(config: TrainingConfig) -> Dict:
    """Main training function"""
    print("\n" + "=" * 60)
    print("AGE ESTIMATION TRAINING")
    print("=" * 60)
    
    # Setup device (CPU only)
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_processed_dataloaders(
        processed_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment_train=True
    )
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")
    
    # Create model
    print(f"\nCreating model: {config.model_type}")
    model = create_model(config.model_type)
    model.to(device)
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    # Loss function (L1 = MAE, more robust to outliers)
    criterion = nn.L1Loss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler (verbose removed - deprecated in PyTorch 2.x)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.early_stop_patience)
    
    # Tracking
    best_val_mae = float('inf')
    history = []
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nTraining for up to {config.epochs} epochs...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_mae, val_rmse = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Create metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_mae=train_mae,
            val_loss=val_loss,
            val_mae=val_mae,
            val_rmse=val_rmse,
            lr=current_lr,
            epoch_time=epoch_time
        )
        history.append(asdict(metrics))
        
        # Print progress
        is_best = val_mae < best_val_mae
        best_marker = " ★" if is_best else ""
        print(f"Epoch {epoch:3d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} | "
              f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f} | "
              f"LR: {current_lr:.6f} | {epoch_time:.1f}s{best_marker}")
        
        # Save best model
        if is_best:
            best_val_mae = val_mae
            save_checkpoint(
                model, optimizer, epoch, metrics,
                checkpoint_dir / 'best_checkpoint.pt',
                is_best=True
            )
        
        # Periodic checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            )
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n  ⚠ Early stopping triggered after {epoch} epochs")
            break
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    
    # Load best model for testing
    print("\nLoading best model for test evaluation...")
    best_weights_path = checkpoint_dir / 'best_model.pt'
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_test(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  MAE:             {test_metrics['mae']:.2f} years")
    print(f"  RMSE:            {test_metrics['rmse']:.2f} years")
    print(f"  Std:             {test_metrics['std']:.2f} years")
    print(f"  Median Error:    {test_metrics['median_error']:.2f} years")
    print(f"  Max Error:       {test_metrics['max_error']:.2f} years")
    print(f"  Within ±5 yrs:   {test_metrics['within_5_years']:.1f}%")
    print(f"  Within ±10 yrs:  {test_metrics['within_10_years']:.1f}%")
    print("=" * 60)
    
    # Save training history
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'config': asdict(config),
        'history': history,
        'test_metrics': test_metrics,
        'best_val_mae': best_val_mae,
        'total_time_seconds': total_time,
        'final_epoch': len(history)
    }
    
    results_path = log_dir / f'training_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Also save final weights with timestamp
    final_weights_path = checkpoint_dir / f'age_model_{timestamp}.pt'
    torch.save(model.state_dict(), final_weights_path)
    print(f"Final weights saved to: {final_weights_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train age estimation model"
    )
    parser.add_argument(
        '--data-dir',
        default='data/processed/utkface',
        help='Path to processed data'
    )
    parser.add_argument(
        '--model',
        default='light',
        choices=['light', 'v2'],
        help='Model type'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run (5 epochs)'
    )
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        model_type=args.model,
        epochs=5 if args.quick else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    train(config)


if __name__ == "__main__":
    main()
