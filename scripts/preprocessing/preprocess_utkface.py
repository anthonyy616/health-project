"""
UTKFace Dataset Preprocessing Script (Memory-Efficient Version)
================================================================
Processes raw UTKFace images into a clean, training-ready format.

This version saves images as individual files to avoid memory issues.
Creates a manifest file with paths and labels for fast loading.

Operations:
1. Validates all images (readable, correct format)
2. Resizes to consistent dimensions (224x224)
3. Normalizes filenames for consistency
4. Creates train/val/test splits with manifest files
5. Saves processed images to organized folder structure

Usage:
    python scripts/preprocess_utkface.py
    python scripts/preprocess_utkface.py --size 128
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
import logging
from dataclasses import dataclass, asdict
from tqdm import tqdm
import random
import sys
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing"""
    raw_dir: str = "data/raw/utkface"
    processed_dir: str = "data/processed/utkface"
    image_size: Tuple[int, int] = (224, 224)
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    max_age: int = 100  # Filter out ages > 100 (likely errors)


def parse_utkface_filename(filename: str) -> Optional[Dict]:
    """Parse UTKFace filename to extract labels."""
    name = filename
    for ext in ['.chip.jpg', '.jpg', '.jpeg', '.png']:
        name = name.replace(ext, '')
    
    parts = name.split('_')
    if len(parts) < 3:
        return None
    
    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        
        if not (0 <= age <= 120):
            return None
        if gender not in (0, 1):
            return None
        if race not in (0, 1, 2, 3, 4):
            return None
        
        return {
            'age': age,
            'gender': gender,
            'race': race
        }
    except (ValueError, IndexError):
        return None


def process_and_save_image(
    src_path: Path,
    dst_path: Path,
    target_size: Tuple[int, int]
) -> bool:
    """Process a single image and save to destination."""
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Save (keep as BGR for cv2, convert during loading)
        cv2.imwrite(str(dst_path), img)
        return True
        
    except Exception as e:
        logger.debug(f"Failed to process {src_path}: {e}")
        return False


def create_splits(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Dict[str, List[Dict]]:
    """Create stratified train/val/test splits."""
    random.seed(seed)
    
    # Define age bins for stratification
    age_bins = [
        (0, 12, 'child'),
        (13, 19, 'teen'),
        (20, 35, 'young_adult'),
        (36, 55, 'middle_aged'),
        (56, 120, 'senior')
    ]
    
    # Group samples by age bin
    binned_samples = {name: [] for _, _, name in age_bins}
    
    for sample in samples:
        age = sample['age']
        for min_age, max_age, name in age_bins:
            if min_age <= age <= max_age:
                binned_samples[name].append(sample)
                break
    
    splits = {'train': [], 'val': [], 'test': []}
    
    # Split each bin proportionally
    for bin_name, bin_samples in binned_samples.items():
        random.shuffle(bin_samples)
        n = len(bin_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits['train'].extend(bin_samples[:n_train])
        splits['val'].extend(bin_samples[n_train:n_train + n_val])
        splits['test'].extend(bin_samples[n_train + n_val:])
    
    # Shuffle each split
    for split in splits.values():
        random.shuffle(split)
    
    return splits


def preprocess_dataset(config: PreprocessConfig) -> Dict:
    """Main preprocessing pipeline."""
    raw_dir = Path(config.raw_dir)
    processed_dir = Path(config.processed_dir)
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    # Clean and recreate output directory
    if processed_dir.exists():
        logger.info(f"Cleaning existing processed directory: {processed_dir}")
        shutil.rmtree(processed_dir)
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (processed_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    all_files = []
    for ext in image_extensions:
        all_files.extend(raw_dir.rglob(f'*{ext}'))
    
    logger.info(f"Found {len(all_files)} image files in {raw_dir}")
    
    # First pass: collect valid samples
    logger.info("Scanning for valid images...")
    valid_samples = []
    
    for img_path in tqdm(all_files, desc="Scanning"):
        parsed = parse_utkface_filename(img_path.name)
        if parsed is None:
            continue
        
        if parsed['age'] > config.max_age:
            continue
        
        # Check image is readable
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        valid_samples.append({
            'original_path': str(img_path),
            'age': parsed['age'],
            'gender': parsed['gender'],
            'race': parsed['race']
        })
    
    logger.info(f"Found {len(valid_samples)} valid samples")
    
    # Create splits
    logger.info("Creating train/val/test splits...")
    splits = create_splits(
        valid_samples,
        config.train_split,
        config.val_split,
        config.seed
    )
    
    # Process and save images for each split
    stats = {
        'total_processed': 0,
        'train_count': 0,
        'val_count': 0,
        'test_count': 0,
        'age_distribution': {}
    }
    
    for split_name, samples in splits.items():
        logger.info(f"Processing {split_name} split ({len(samples)} images)...")
        split_dir = processed_dir / split_name
        manifest = []
        
        for i, sample in enumerate(tqdm(samples, desc=f"{split_name}")):
            # Create unique filename
            new_filename = f"{i:06d}_{sample['age']}_{sample['gender']}_{sample['race']}.jpg"
            dst_path = split_dir / new_filename
            
            # Process and save
            success = process_and_save_image(
                Path(sample['original_path']),
                dst_path,
                config.image_size
            )
            
            if success:
                manifest.append({
                    'filename': new_filename,
                    'age': sample['age'],
                    'gender': sample['gender'],
                    'race': sample['race']
                })
                stats['total_processed'] += 1
                
                # Track age distribution
                age_bin = f"{(sample['age'] // 10) * 10}s"
                stats['age_distribution'][age_bin] = stats['age_distribution'].get(age_bin, 0) + 1
        
        # Save manifest
        manifest_path = processed_dir / f"{split_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'count': len(manifest),
                'image_size': list(config.image_size),
                'samples': manifest
            }, f, indent=2)
        
        stats[f'{split_name}_count'] = len(manifest)
        logger.info(f"Saved {split_name} manifest: {len(manifest)} samples")
    
    # Save config and stats
    with open(processed_dir / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    with open(processed_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def print_summary(stats: Dict, config: PreprocessConfig):
    """Print processing summary."""
    print("\n" + "=" * 60)
    print("UTKFACE PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {config.processed_dir}")
    print(f"Image size: {config.image_size[0]}x{config.image_size[1]}")
    
    print(f"\n📁 Split Sizes:")
    print(f"   Train: {stats['train_count']:,} ({config.train_split*100:.0f}%)")
    print(f"   Val:   {stats['val_count']:,} ({config.val_split*100:.0f}%)")
    print(f"   Test:  {stats['test_count']:,} ({config.test_split*100:.0f}%)")
    print(f"   Total: {stats['total_processed']:,}")
    
    print(f"\n👤 Age Distribution:")
    if stats['age_distribution']:
        sorted_bins = sorted(stats['age_distribution'].items(), 
                            key=lambda x: int(x[0].replace('s', '')))
        for age_bin, count in sorted_bins:
            pct = count / stats['total_processed'] * 100
            bar = "█" * int(pct / 2)
            print(f"   {age_bin:5} {count:5,} ({pct:5.1f}%) {bar}")
    
    print("\n" + "=" * 60)
    print("✅ Ready for training!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess UTKFace dataset for age estimation"
    )
    parser.add_argument(
        '--raw-dir', 
        default='data/raw/utkface',
        help='Path to raw UTKFace data'
    )
    parser.add_argument(
        '--processed-dir',
        default='data/processed/utkface',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Target image size (square)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splits'
    )
    
    args = parser.parse_args()
    
    config = PreprocessConfig(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        image_size=(args.size, args.size),
        seed=args.seed
    )
    
    logger.info("Starting UTKFace preprocessing (memory-efficient version)...")
    stats = preprocess_dataset(config)
    print_summary(stats, config)


if __name__ == "__main__":
    main()
