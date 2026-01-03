"""
Processed UTKFace Dataset Loader for Age Estimation
=====================================================
Loads preprocessed images from the processed data folder.
Uses manifest files for fast loading without directory scanning.

Compatible with PyTorch DataLoader.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
import json
import random
import logging

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about the dataset"""
    total_images: int
    age_distribution: Dict[str, int]
    gender_distribution: Dict[str, int]
    race_distribution: Dict[str, int]
    min_age: int
    max_age: int
    mean_age: float


class ProcessedUTKFaceDataset(Dataset):
    """
    PyTorch Dataset for preprocessed UTKFace images.
    
    Loads from processed folder with manifest files for fast initialization.
    
    Usage:
        dataset = ProcessedUTKFaceDataset(split='train')
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    GENDER_MAP = {0: 'male', 1: 'female'}
    RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
    
    def __init__(
        self,
        processed_dir: str = "data/processed/utkface",
        split: str = "train",
        transform: Optional[Callable] = None,
        augment: bool = False,
        normalize: bool = True
    ):
        """
        Initialize the processed UTKFace dataset.
        
        Args:
            processed_dir: Path to processed data directory
            split: One of 'train', 'val', 'test'
            transform: Optional custom transform function
            augment: Whether to apply data augmentation (only for train)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.transform = transform
        self.augment = augment and (split == 'train')  # Only augment training data
        self.normalize = normalize
        
        # Load manifest
        manifest_path = self.processed_dir / f"{split}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                "Run 'python scripts/preprocess_utkface.py' first."
            )
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.samples = manifest['samples']
        self.image_size = tuple(manifest['image_size'])
        self.split_dir = self.processed_dir / split
        
        logger.info(f"Loaded {len(self.samples)} {split} samples from manifest")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (image_tensor, age)
        """
        sample = self.samples[idx]
        img_path = self.split_dir / sample['filename']
        
        # Load image (already resized during preprocessing)
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if enabled
        if self.augment:
            image = self._apply_augmentation(image)
        
        # Apply custom transform or default processing
        if self.transform:
            image = self.transform(image)
        else:
            image = image.astype(np.float32)
            if self.normalize:
                image = image / 255.0
            # HWC -> CHW for PyTorch
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        
        return image, sample['age']
    
    def get_sample_with_metadata(self, idx: int) -> Dict:
        """Get sample with all metadata."""
        sample = self.samples[idx]
        image, age = self[idx]
        
        return {
            'image': image,
            'age': age,
            'gender': sample['gender'],
            'race': sample['race'],
            'filename': sample['filename'],
            'gender_name': self.GENDER_MAP[sample['gender']],
            'race_name': self.RACE_MAP[sample['race']]
        }
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations for training."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return image
    
    def get_statistics(self) -> DatasetStats:
        """Calculate dataset statistics."""
        ages = [s['age'] for s in self.samples]
        genders = [s['gender'] for s in self.samples]
        races = [s['race'] for s in self.samples]
        
        # Age distribution by decades
        age_dist = {}
        for age in ages:
            bin_name = f"{(age // 10) * 10}s"
            age_dist[bin_name] = age_dist.get(bin_name, 0) + 1
        
        # Gender distribution
        gender_dist = {
            'male': sum(1 for g in genders if g == 0),
            'female': sum(1 for g in genders if g == 1)
        }
        
        # Race distribution
        race_dist = {}
        for race_id, race_name in self.RACE_MAP.items():
            race_dist[race_name] = sum(1 for r in races if r == race_id)
        
        return DatasetStats(
            total_images=len(self.samples),
            age_distribution=age_dist,
            gender_distribution=gender_dist,
            race_distribution=race_dist,
            min_age=min(ages),
            max_age=max(ages),
            mean_age=np.mean(ages)
        )
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'=' * 60}")
        print(f"PROCESSED UTKFACE DATASET - {self.split.upper()}")
        print(f"{'=' * 60}")
        print(f"Total images: {stats.total_images:,}")
        print(f"Image size: {self.image_size}")
        print(f"Age range: {stats.min_age} - {stats.max_age} years")
        print(f"Mean age: {stats.mean_age:.1f} years")
        
        print("\nAge Distribution:")
        sorted_bins = sorted(stats.age_distribution.items(),
                            key=lambda x: int(x[0].replace('s', '')))
        for bin_name, count in sorted_bins:
            pct = count / stats.total_images * 100
            bar = "█" * int(pct / 2)
            print(f"  {bin_name:5} {count:5,} ({pct:5.1f}%) {bar}")
        
        print(f"{'=' * 60}\n")


def get_processed_dataloaders(
    processed_dir: str = "data/processed/utkface",
    batch_size: int = 32,
    num_workers: int = 0,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from processed data.
    
    Args:
        processed_dir: Path to processed data directory
        batch_size: Batch size for all loaders
        num_workers: Number of workers for data loading (0 for Windows)
        augment_train: Whether to apply augmentation to training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = ProcessedUTKFaceDataset(
        processed_dir=processed_dir,
        split='train',
        augment=augment_train
    )
    
    val_dataset = ProcessedUTKFaceDataset(
        processed_dir=processed_dir,
        split='val',
        augment=False
    )
    
    test_dataset = ProcessedUTKFaceDataset(
        processed_dir=processed_dir,
        split='test',
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created loaders - Train: {len(train_loader)} batches, "
                f"Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def get_balanced_dataloaders(
    processed_dir: str = "data/processed/utkface",
    batch_size: int = 32,
    num_workers: int = 0,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders with age-balanced sampling for training.
    
    Uses WeightedRandomSampler to give equal probability to all age bins,
    fixing the dataset imbalance (too many 20-30 year olds).
    
    Args:
        processed_dir: Path to processed data directory
        batch_size: Batch size for all loaders
        num_workers: Number of workers for data loading
        augment_train: Whether to apply augmentation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import WeightedRandomSampler
    
    train_dataset = ProcessedUTKFaceDataset(
        processed_dir=processed_dir,
        split='train',
        augment=augment_train
    )
    
    val_dataset = ProcessedUTKFaceDataset(
        processed_dir=processed_dir,
        split='val',
        augment=False
    )
    
    test_dataset = ProcessedUTKFaceDataset(
        processed_dir=processed_dir,
        split='test',
        augment=False
    )
    
    # Calculate sample weights based on age bins
    # Age bins: [0-10, 10-20, 20-30, ..., 90-100]
    age_bins = {}
    for sample in train_dataset.samples:
        age_bin = sample['age'] // 10
        age_bins[age_bin] = age_bins.get(age_bin, 0) + 1
    
    # Calculate weight for each bin (inverse frequency)
    total_samples = len(train_dataset.samples)
    bin_weights = {
        bin_id: total_samples / count 
        for bin_id, count in age_bins.items()
    }
    
    # Assign weight to each sample
    sample_weights = []
    for sample in train_dataset.samples:
        age_bin = sample['age'] // 10
        sample_weights.append(bin_weights[age_bin])
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Log rebalancing info
    logger.info("Age-balanced sampling enabled:")
    for bin_id in sorted(age_bins.keys()):
        count = age_bins[bin_id]
        weight = bin_weights[bin_id]
        logger.info(f"  Age {bin_id*10}-{(bin_id+1)*10}: {count:5} samples, weight: {weight:.2f}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created balanced loaders - Train: {len(train_loader)} batches")
    
    return train_loader, val_loader, test_loader
