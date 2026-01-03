"""
UTKFace Dataset Preprocessing with Face Alignment
==================================================
Enhanced preprocessing that aligns faces using MediaPipe landmarks.

Improvements over basic version:
1. Face detection and alignment using MediaPipe
2. Eyes aligned horizontally  
3. Consistent face cropping
4. Better normalization for training

Usage:
    python scripts/preprocessing/preprocess_aligned.py
    python scripts/preprocessing/preprocess_aligned.py --output-dir data/processed/utkface_aligned
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
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AlignedPreprocessConfig:
    """Configuration for aligned preprocessing"""
    raw_dir: str = "data/raw/utkface"
    processed_dir: str = "data/processed/utkface_aligned"
    image_size: Tuple[int, int] = (224, 224)
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    max_age: int = 100
    # Face alignment settings
    face_padding: float = 0.3  # Padding around detected face
    min_face_size: int = 50   # Minimum face size to process


class FaceAligner:
    """
    Face alignment using MediaPipe Tasks API (FaceLandmarker).
    
    Aligns faces by:
    1. Detecting face landmarks
    2. Finding eye positions
    3. Rotating to make eyes horizontal
    4. Cropping with consistent padding
    """
    
    # Landmark indices for eyes (from FaceLandmarker)
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    
    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    def __init__(self):
        # Download model if needed
        self._ensure_model()
        
        # Initialize MediaPipe FaceLandmarker
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python import BaseOptions
        
        base_options = BaseOptions(model_asset_path=str(self.MODEL_PATH))
        options = FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
    
    def _ensure_model(self):
        """Download the face landmark model if not present."""
        if self.MODEL_PATH.exists():
            return
        
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        import urllib.request
        logger.info(f"Downloading face landmarker model...")
        urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
        logger.info(f"Model saved to {self.MODEL_PATH}")
    
    def align_face(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        padding: float = 0.3
    ) -> Optional[np.ndarray]:
        """
        Align and crop face from image.
        
        Args:
            image: BGR image from OpenCV
            target_size: Output size (width, height)
            padding: Padding around face as fraction
            
        Returns:
            Aligned face image or None if detection failed
        """
        import mediapipe as mp
        
        h, w = image.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect landmarks
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        # Get first face landmarks
        landmarks = result.face_landmarks[0]
        
        # Get eye positions (landmarks are normalized 0-1)
        left_eye = landmarks[self.LEFT_EYE_OUTER]
        right_eye = landmarks[self.RIGHT_EYE_OUTER]
        
        left_eye_pt = np.array([left_eye.x * w, left_eye.y * h])
        right_eye_pt = np.array([right_eye.x * w, right_eye.y * h])
        
        # Calculate rotation angle
        dY = right_eye_pt[1] - left_eye_pt[1]
        dX = right_eye_pt[0] - left_eye_pt[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate eye center
        eye_center = (left_eye_pt + right_eye_pt) / 2
        
        # Calculate face bounding box from landmarks
        all_x = [lm.x * w for lm in landmarks]
        all_y = [lm.y * h for lm in landmarks]
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        face_width = x_max - x_min
        face_height = y_max - y_min
        
        # Add padding
        pad_x = face_width * padding
        pad_y = face_height * padding
        
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Transform bounding box corners
        corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
        
        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])
        rotated_corners = M @ corners_h.T
        rotated_corners = rotated_corners.T
        
        new_x_min = max(0, int(rotated_corners[:, 0].min()))
        new_x_max = min(w, int(rotated_corners[:, 0].max()))
        new_y_min = max(0, int(rotated_corners[:, 1].min()))
        new_y_max = min(h, int(rotated_corners[:, 1].max()))
        
        # Crop face
        face = rotated[new_y_min:new_y_max, new_x_min:new_x_max]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        aligned = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
        
        return aligned
    
    def close(self):
        """Release resources"""
        self.landmarker.close()


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
        
        return {'age': age, 'gender': gender, 'race': race}
    except (ValueError, IndexError):
        return None


def create_stratified_splits(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Dict[str, List[Dict]]:
    """Create stratified train/val/test splits by age bins."""
    random.seed(seed)
    
    age_bins = [
        (0, 12, 'child'),
        (13, 19, 'teen'),
        (20, 35, 'young_adult'),
        (36, 55, 'middle_aged'),
        (56, 120, 'senior')
    ]
    
    binned_samples = {name: [] for _, _, name in age_bins}
    
    for sample in samples:
        age = sample['age']
        for min_age, max_age, name in age_bins:
            if min_age <= age <= max_age:
                binned_samples[name].append(sample)
                break
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for bin_name, bin_samples in binned_samples.items():
        random.shuffle(bin_samples)
        n = len(bin_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits['train'].extend(bin_samples[:n_train])
        splits['val'].extend(bin_samples[n_train:n_train + n_val])
        splits['test'].extend(bin_samples[n_train + n_val:])
    
    for split in splits.values():
        random.shuffle(split)
    
    return splits


def preprocess_with_alignment(config: AlignedPreprocessConfig) -> Dict:
    """Main preprocessing pipeline with face alignment."""
    raw_dir = Path(config.raw_dir)
    processed_dir = Path(config.processed_dir)
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    # Clean and recreate output directory
    if processed_dir.exists():
        logger.info(f"Cleaning existing directory: {processed_dir}")
        shutil.rmtree(processed_dir)
    
    for split in ['train', 'val', 'test']:
        (processed_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Initialize face aligner
    aligner = FaceAligner()
    
    # Find all image files
    all_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        all_files.extend(raw_dir.rglob(f'*{ext}'))
    
    logger.info(f"Found {len(all_files)} image files")
    
    # First pass: process images and collect valid samples
    logger.info("Processing and aligning faces...")
    valid_samples = []
    failed_count = 0
    
    for img_path in tqdm(all_files, desc="Aligning"):
        parsed = parse_utkface_filename(img_path.name)
        if parsed is None:
            continue
        
        if parsed['age'] > config.max_age:
            continue
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            failed_count += 1
            continue
        
        # Align face
        aligned = aligner.align_face(
            img,
            target_size=config.image_size,
            padding=config.face_padding
        )
        
        if aligned is None:
            failed_count += 1
            continue
        
        valid_samples.append({
            'original_path': str(img_path),
            'aligned_image': aligned,
            'age': parsed['age'],
            'gender': parsed['gender'],
            'race': parsed['race']
        })
    
    aligner.close()
    
    logger.info(f"Successfully aligned {len(valid_samples)} faces")
    logger.info(f"Failed to align {failed_count} images")
    
    # Create splits
    logger.info("Creating train/val/test splits...")
    splits = create_stratified_splits(
        valid_samples,
        config.train_split,
        config.val_split,
        config.seed
    )
    
    # Save processed images
    stats = {
        'total_processed': 0,
        'train_count': 0,
        'val_count': 0,
        'test_count': 0,
        'failed_alignment': failed_count,
        'age_distribution': {}
    }
    
    for split_name, samples in splits.items():
        logger.info(f"Saving {split_name} split ({len(samples)} images)...")
        split_dir = processed_dir / split_name
        manifest = []
        
        for i, sample in enumerate(tqdm(samples, desc=f"{split_name}")):
            new_filename = f"{i:06d}_{sample['age']}_{sample['gender']}_{sample['race']}.jpg"
            dst_path = split_dir / new_filename
            
            # Save aligned image
            cv2.imwrite(str(dst_path), sample['aligned_image'])
            
            manifest.append({
                'filename': new_filename,
                'age': sample['age'],
                'gender': sample['gender'],
                'race': sample['race']
            })
            stats['total_processed'] += 1
            
            age_bin = f"{(sample['age'] // 10) * 10}s"
            stats['age_distribution'][age_bin] = stats['age_distribution'].get(age_bin, 0) + 1
        
        # Save manifest
        manifest_path = processed_dir / f"{split_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'count': len(manifest),
                'image_size': list(config.image_size),
                'aligned': True,
                'samples': manifest
            }, f, indent=2)
        
        stats[f'{split_name}_count'] = len(manifest)
    
    # Save config and stats
    with open(processed_dir / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    with open(processed_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def print_summary(stats: Dict, config: AlignedPreprocessConfig):
    """Print processing summary."""
    print("\n" + "=" * 60)
    print("ALIGNED UTKFACE PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {config.processed_dir}")
    print(f"Image size: {config.image_size[0]}x{config.image_size[1]}")
    print(f"Face padding: {config.face_padding}")
    
    print(f"\n📁 Split Sizes:")
    print(f"   Train: {stats['train_count']:,}")
    print(f"   Val:   {stats['val_count']:,}")
    print(f"   Test:  {stats['test_count']:,}")
    print(f"   Total: {stats['total_processed']:,}")
    print(f"   Failed alignment: {stats['failed_alignment']:,}")
    
    print(f"\n👤 Age Distribution:")
    sorted_bins = sorted(
        stats['age_distribution'].items(),
        key=lambda x: int(x[0].replace('s', ''))
    )
    for age_bin, count in sorted_bins:
        pct = count / stats['total_processed'] * 100
        bar = "█" * int(pct / 2)
        print(f"   {age_bin:5} {count:5,} ({pct:5.1f}%) {bar}")
    
    print("\n" + "=" * 60)
    print("✅ Aligned dataset ready for training!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess UTKFace with face alignment"
    )
    parser.add_argument(
        '--raw-dir',
        default='data/raw/utkface',
        help='Path to raw UTKFace data'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/utkface_aligned',
        help='Output directory'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Target image size'
    )
    parser.add_argument(
        '--padding',
        type=float,
        default=0.3,
        help='Face padding ratio'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    config = AlignedPreprocessConfig(
        raw_dir=args.raw_dir,
        processed_dir=args.output_dir,
        image_size=(args.size, args.size),
        face_padding=args.padding,
        seed=args.seed
    )
    
    logger.info("Starting aligned preprocessing...")
    stats = preprocess_with_alignment(config)
    print_summary(stats, config)


if __name__ == "__main__":
    main()
