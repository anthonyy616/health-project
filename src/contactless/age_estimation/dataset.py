"""
UTKFace Dataset Loader for Age Estimation
Handles loading, preprocessing, and splitting the UTKFace dataset.
Compatible with PyTorch DataLoader.
"""

import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
import random
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about the dataset"""
    total_images: int
    valid_images: int
    invalid_images: int
    age_distribution: Dict[str, int]  # Age bins -> count
    gender_distribution: Dict[str, int]  # Gender -> count
    race_distribution: Dict[str, int]  # Race -> count
    min_age: int
    max_age: int
    mean_age: float


class UTKFaceDataset(Dataset):
    """
    PyTorch Dataset for UTKFace.
    
    UTKFace naming format: [age]_[gender]_[race]_[date&time].jpg
    - age: integer from 0 to 116
    - gender: 0 (male) or 1 (female)
    - race: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
    
    Usage:
        dataset = UTKFaceDataset(root_dir="data/raw/utkface")
        train_loader = DataLoader(dataset.get_train_split(), batch_size=32)
    """
    
    # Age bin ranges for stratified sampling and analysis
    AGE_BINS = {
        'infant': (0, 2),
        'child': (3, 12),
        'teenager': (13, 19),
        'young_adult': (20, 35),
        'middle_aged': (36, 55),
        'senior': (56, 116)
    }
    
    GENDER_MAP = {0: 'male', 1: 'female'}
    RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
    
    def __init__(
        self,
        root_dir: str = "data/raw/utkface",
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        augment: bool = False,
        filter_age_range: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ):
        """
        Initialize the UTKFace dataset.
        
        Args:
            root_dir: Path to UTKFace directory
            image_size: Target size for images (H, W)
            transform: Optional custom transform function
            augment: Whether to apply data augmentation
            filter_age_range: Optional (min_age, max_age) to filter samples
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.augment = augment
        self.filter_age_range = filter_age_range
        self.normalize = normalize
        
        # Find all images
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root_dir}. "
                           "Please check the dataset path.")
        
        logger.info(f"Loaded {len(self.samples)} valid samples from UTKFace")
    
    def _load_samples(self) -> List[Dict]:
        """Scan directory and parse all valid image files"""
        samples = []
        invalid_count = 0
        
        # Search recursively in all subdirectories
        image_extensions = {'.jpg', '.jpeg', '.png', '.chip.jpg'}
        
        for img_path in self.root_dir.rglob('*'):
            if not img_path.is_file():
                continue
                
            # Check if it's an image file
            suffix = ''.join(img_path.suffixes).lower()
            if not any(suffix.endswith(ext) for ext in image_extensions):
                continue
            
            # Parse filename for labels
            parsed = self._parse_filename(img_path.name)
            if parsed is None:
                invalid_count += 1
                continue
            
            age, gender, race = parsed
            
            # Apply age filter if specified
            if self.filter_age_range:
                min_age, max_age = self.filter_age_range
                if not (min_age <= age <= max_age):
                    continue
            
            samples.append({
                'path': str(img_path),
                'age': age,
                'gender': gender,
                'race': race
            })
        
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} files with invalid naming format")
        
        return samples
    
    def _parse_filename(self, filename: str) -> Optional[Tuple[int, int, int]]:
        """
        Parse UTKFace filename to extract age, gender, race.
        
        Format: [age]_[gender]_[race]_[date].jpg
        Returns: (age, gender, race) or None if invalid
        """
        # Remove .chip.jpg suffix if present
        name = filename.replace('.chip.jpg', '.jpg')
        name = name.replace('.jpg', '')
        
        parts = name.split('_')
        if len(parts) < 3:
            return None
        
        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
            
            # Validate ranges
            if not (0 <= age <= 120):
                return None
            if gender not in (0, 1):
                return None
            if race not in (0, 1, 2, 3, 4):
                return None
            
            return age, gender, race
            
        except (ValueError, IndexError):
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (image_tensor, age)
            - image_tensor: (C, H, W) normalized tensor
            - age: integer age label
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['path'])
        if image is None:
            raise IOError(f"Failed to load image: {sample['path']}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Apply augmentation if enabled
        if self.augment:
            image = self._apply_augmentation(image)
        
        # Apply custom transform if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = image.astype(np.float32)
            if self.normalize:
                image = image / 255.0
            # HWC -> CHW for PyTorch
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        
        age = sample['age']
        
        return image, age
    
    def get_sample_with_metadata(self, idx: int) -> Dict:
        """Get sample with all metadata (for analysis)"""
        sample = self.samples[idx]
        image, age = self[idx]
        
        return {
            'image': image,
            'age': age,
            'gender': sample['gender'],
            'race': sample['race'],
            'path': sample['path'],
            'gender_name': self.GENDER_MAP[sample['gender']],
            'race_name': self.RACE_MAP[sample['race']]
        }
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations for training"""
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
        """Calculate and return dataset statistics"""
        ages = [s['age'] for s in self.samples]
        genders = [s['gender'] for s in self.samples]
        races = [s['race'] for s in self.samples]
        
        # Age distribution by bins
        age_dist = {}
        for bin_name, (min_age, max_age) in self.AGE_BINS.items():
            count = sum(1 for a in ages if min_age <= a <= max_age)
            age_dist[bin_name] = count
        
        # Gender distribution
        gender_dist = {
            self.GENDER_MAP[0]: sum(1 for g in genders if g == 0),
            self.GENDER_MAP[1]: sum(1 for g in genders if g == 1)
        }
        
        # Race distribution
        race_dist = {}
        for race_id, race_name in self.RACE_MAP.items():
            race_dist[race_name] = sum(1 for r in races if r == race_id)
        
        return DatasetStats(
            total_images=len(self.samples),
            valid_images=len(self.samples),
            invalid_images=0,  # Already filtered
            age_distribution=age_dist,
            gender_distribution=gender_dist,
            race_distribution=race_dist,
            min_age=min(ages),
            max_age=max(ages),
            mean_age=np.mean(ages)
        )
    
    def print_statistics(self):
        """Print dataset statistics in a formatted way"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("UTKFACE DATASET STATISTICS")
        print("=" * 60)
        print(f"Total images: {stats.total_images:,}")
        print(f"Age range: {stats.min_age} - {stats.max_age} years")
        print(f"Mean age: {stats.mean_age:.1f} years")
        
        print("\nAge Distribution:")
        for bin_name, count in stats.age_distribution.items():
            pct = count / stats.total_images * 100
            bar = "█" * int(pct / 2)
            print(f"  {bin_name:15} {count:5,} ({pct:5.1f}%) {bar}")
        
        print("\nGender Distribution:")
        for gender, count in stats.gender_distribution.items():
            pct = count / stats.total_images * 100
            print(f"  {gender:10} {count:5,} ({pct:5.1f}%)")
        
        print("\nRace Distribution:")
        for race, count in stats.race_distribution.items():
            pct = count / stats.total_images * 100
            print(f"  {race:10} {count:5,} ({pct:5.1f}%)")
        
        print("=" * 60 + "\n")


def get_age_train_val_test_loaders(
    root_dir: str = "data/raw/utkface",
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders with stratified sampling.
    
    Args:
        root_dir: Path to UTKFace directory
        batch_size: Batch size for training
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        image_size: Target image size
        num_workers: Number of workers for loading (0 for Windows)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load full dataset (no augmentation for splitting)
    full_dataset = UTKFaceDataset(
        root_dir=root_dir,
        image_size=image_size,
        augment=False,
        normalize=True
    )
    
    # Create indices
    n_samples = len(full_dataset)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    # Split indices
    n_train = int(train_split * n_samples)
    n_val = int(val_split * n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(
        UTKFaceDataset(root_dir=root_dir, image_size=image_size, augment=True),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        UTKFaceDataset(root_dir=root_dir, image_size=image_size, augment=False),
        val_indices
    )
    test_dataset = torch.utils.data.Subset(
        UTKFaceDataset(root_dir=root_dir, image_size=image_size, augment=False),
        test_indices
    )
    
    # Create data loaders
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
    
    logger.info(f"Dataset splits - Train: {len(train_indices)}, "
                f"Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_loader, val_loader, test_loader
