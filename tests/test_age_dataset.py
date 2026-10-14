"""
Tests for UTKFace Dataset Loaders
Run with: python -m pytest tests/test_age_dataset.py -v
Or standalone: python tests/test_age_dataset.py

Real-split tests skip explicitly (pytest.skip) when the processed UTKFace
splits are missing. Dataset-INDEPENDENT behaviour (manifest parsing, image
loading, normalization, metadata mapping, batching, augmentation gating)
is covered by a deterministic synthetic processed-directory fixture so this
coverage does not depend on the (deleted) real dataset.
"""

import json
import numpy as np
import pytest
from pathlib import Path

# Add project root to path


class TestProcessedDataset:
    """Test cases for ProcessedUTKFaceDataset (primary dataset loader)

    These tests are valid only when the processed UTKFace splits are present
    on disk. When they are missing, the suite skips explicitly rather than
    failing with confusing FileNotFoundError / manifest errors.
    """

    @pytest.fixture
    def processed_dir_available(self):
        """Skip the whole processed-dataset class when the processed data is absent."""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset

        for split in ("train", "val", "test"):
            manifest = Path("data/processed/utkface") / f"{split}_manifest.json"
            if not manifest.exists():
                pytest.skip(
                    f"Processed UTKFace data not available: missing "
                    f"data/processed/utkface/{split}_manifest.json. "
                    f"Run scripts/preprocess_utkface.py first."
                )
        return True

    @pytest.fixture
    def dataset(self, processed_dir_available):
        """Create dataset instance for testing"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        return ProcessedUTKFaceDataset(
            processed_dir="data/processed/utkface",
            split="train",
            augment=False
        )
    
    def test_dataset_loads(self, dataset):
        """Test that dataset loads without errors"""
        assert len(dataset) > 0, "Dataset should not be empty"
        print(f"✅ Dataset loaded with {len(dataset)} samples")
    
    # test_sample_shape and test_image_normalization removed - depend on image files
    # that are no longer on disk (deleted by user).
    
    def test_age_range(self, dataset):
        """Test that ages are in valid range"""
        ages = [dataset.samples[i]['age'] for i in range(min(100, len(dataset)))]
        
        assert all(0 <= age <= 120 for age in ages), \
            "All ages should be in range [0, 120]"
        print(f"✅ Age range in sample: {min(ages)} - {max(ages)}")
    
    def test_statistics(self, dataset):
        """Test statistics calculation"""
        stats = dataset.get_statistics()
        
        assert stats.total_images > 0
        assert stats.min_age >= 0
        assert stats.max_age <= 120
        assert stats.mean_age > 0
        print(f"✅ Statistics: {stats.total_images} images, "
              f"mean age: {stats.mean_age:.1f}")
    
    # test_metadata_retrieval removed - depends on image files no longer on disk


class TestDataLoaders:
    """Test cases for data loader creation"""
    
    def test_loader_creation(self):
        """Test that data loaders are created correctly"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset, get_processed_dataloaders
        import pytest
        from pathlib import Path

        # Validate the required split manifests exist before testing loaders
        for split in ("train", "val", "test"):
            manifest = Path("data/processed/utkface") / f"{split}_manifest.json"
            if not manifest.exists():
                pytest.skip(
                    f"Processed UTKFace data not available: missing "
                    f"data/processed/utkface/{split}_manifest.json. "
                    f"Run scripts/preprocess_utkface.py first."
                )

        train_loader, val_loader, test_loader = get_processed_dataloaders(
            processed_dir="data/processed/utkface",
            batch_size=8,
            num_workers=0
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        print(f"✅ Loaders created - Train: {len(train_loader)} batches, "
              f"Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # test_batch_loading removed - depends on image files no longer on disk


class TestAllSplits:
    """Test all data splits"""
    
    def _require_splits(self):
        """Skip when the processed splits are not present on disk."""
        from pathlib import Path
        import pytest

        for split in ("train", "val", "test"):
            manifest = Path("data/processed/utkface") / f"{split}_manifest.json"
            if not manifest.exists():
                pytest.skip(
                    f"Processed UTKFace data not available: missing "
                    f"data/processed/utkface/{split}_manifest.json. "
                    f"Run scripts/preprocess_utkface.py first."
                )

    def test_train_split(self):
        """Test train split loads correctly"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        
        self._require_splits()
        dataset = ProcessedUTKFaceDataset(split='train')
        assert len(dataset) > 40000, f"Expected ~46k train samples, got {len(dataset)}"
        print(f"✅ Train split: {len(dataset)} samples")
    
    def test_val_split(self):
        """Test validation split loads correctly"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        
        self._require_splits()
        dataset = ProcessedUTKFaceDataset(split='val')
        assert len(dataset) > 9000, f"Expected ~10k val samples, got {len(dataset)}"
        print(f"✅ Val split: {len(dataset)} samples")
    
    def test_test_split(self):
        """Test test split loads correctly"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        
        self._require_splits()
        dataset = ProcessedUTKFaceDataset(split='test')
        assert len(dataset) > 9000, f"Expected ~10k test samples, got {len(dataset)}"
        print(f"✅ Test split: {len(dataset)} samples")


class TestSyntheticProcessedFixture:
    """Dataset-independent coverage with a tiny deterministic processed dir.

    Restores coverage removed when the real dataset was deleted: sample
    shape, normalization, metadata retrieval, batch loading and
    augmentation gating are all fixture-based now.
    """

    IMAGE_SIZE = (64, 64)
    N_SAMPLES = 8

    @pytest.fixture
    def synthetic_processed_dir(self, tmp_path):
        """Create a deterministic mini processed-UTKFace layout in tmp."""
        import cv2

        rng = np.random.RandomState(7)
        ages = [1, 5, 18, 25, 34, 47, 62, 78]
        genders = [0, 1] * 4
        races = [0, 1, 2, 3, 4, 0, 1, 2]

        root = tmp_path / "processed"
        samples = []
        for split in ("train", "val", "test"):
            split_dir = root / split
            split_dir.mkdir(parents=True)
            split_samples = []
            for i, age in enumerate(ages):
                filename = f"synthetic_{i:03d}.png"
                # Deterministic gradient image, not zeros: exercises
                # normalization (mean is non-zero) and channel handling.
                base = rng.randint(40, 200, size=(self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3), dtype=np.uint8)
                base = cv2.flip(base, 1 - split_dir.name.count("t") % 2)  # tiny split-dependent variation
                cv2.imwrite(str(split_dir / filename), base)
                split_samples.append({
                    "filename": filename,
                    "age": (age + i) % 80 if split == "train" else age,
                    "gender": genders[i % len(genders)],
                    "race": races[i % len(races)],
                })
            manifest = {
                "samples": split_samples,
                "image_size": list(self.IMAGE_SIZE),
            }
            with open(root / f"{split}_manifest.json", "w") as f:
                json.dump(manifest, f)
        return root

    def _dataset(self, synthetic_processed_dir, split="train", **kwargs):
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        return ProcessedUTKFaceDataset(
            processed_dir=str(synthetic_processed_dir), split=split, **kwargs
        )

    def test_fixtures_load_without_real_dataset(self, synthetic_processed_dir):
        dataset = self._dataset(synthetic_processed_dir)
        assert len(dataset) == self.N_SAMPLES

    def test_sample_shape(self, synthetic_processed_dir):
        """Samples are CHW tensors of the manifest image size."""
        import torch
        dataset = self._dataset(synthetic_processed_dir)
        image, age = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1])
        assert image.dtype == torch.float32
        assert isinstance(age, int)

    def test_image_normalization(self, synthetic_processed_dir):
        """Normalization maps pixels to [0, 1] and is not degenerate."""
        dataset = self._dataset(synthetic_processed_dir)
        image, _ = dataset[3]
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert image.mean() > 0.05, "normalized image must not be all-zeros"

    def test_normalization_disabled(self, synthetic_processed_dir):
        dataset = self._dataset(synthetic_processed_dir, normalize=False)
        image, _ = dataset[0]
        # uint8 scale maintained (0-255 range as float)
        assert image.max() > 1.0 and image.max() <= 255.0

    def test_metadata_retrieval(self, synthetic_processed_dir):
        """Metadata ids map back to the documented gender/race names."""
        dataset = self._dataset(synthetic_processed_dir)
        meta = dataset.get_sample_with_metadata(0)
        assert set(meta) >= {"image", "age", "gender", "race", "filename", "gender_name", "race_name"}
        assert meta["age"] == dataset.samples[0]["age"]
        assert meta["gender_name"] in {"male", "female"}
        assert meta["race_name"] in {"white", "black", "asian", "indian", "others"}
        assert meta["filename"].endswith(".png")

    def test_batch_loading(self, synthetic_processed_dir):
        """DataLoader batches the synthetic fixture correctly."""
        from src.contactless.age_estimation.processed_dataset import get_processed_dataloaders
        train_loader, val_loader, test_loader = get_processed_dataloaders(
            processed_dir=str(synthetic_processed_dir), batch_size=4, num_workers=0
        )
        assert len(train_loader) > 0 and len(val_loader) > 0 and len(test_loader) > 0
        images, ages = next(iter(train_loader))
        assert images.shape == (4, 3, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1])
        assert tuple(ages.shape) == (4,)

    def test_augmentation_gated_to_train_split(self, synthetic_processed_dir):
        """augment=True is forced off for val/test; on for train."""
        train_ds = self._dataset(synthetic_processed_dir, "train", augment=True)
        val_ds = self._dataset(synthetic_processed_dir, "val", augment=True)
        assert train_ds.augment is True
        assert val_ds.augment is False

    def test_augmented_output_stays_uint8_rgb(self, synthetic_processed_dir):
        import random
        random.seed(0)
        dataset = self._dataset(synthetic_processed_dir, "train", augment=True)
        # Call __getitem__ until at least one augmented branch engages,
        # then verify output is still finite uint8-range HWC image.
        for i in range(self.N_SAMPLES):
            image, age = dataset[i]
            image_np, age_val = dataset.get_sample_with_metadata(i)["image"], age
            del image_np
        # Verify through the direct path with a forced augmentation call.
        img = dataset.samples[0]
        import cv2
        raw = cv2.imread(str(dataset.split_dir / img["filename"]))
        augmented = dataset._apply_augmentation(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        assert augmented.dtype == np.uint8
        assert augmented.shape[:2] == self.IMAGE_SIZE
        assert 0 <= augmented.min() <= augmented.max() <= 255

    def test_missing_manifest_raises_file_not_found(self, tmp_path):
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            ProcessedUTKFaceDataset(processed_dir=str(tmp_path / "nowhere"), split="train")

    def test_missing_image_raises_io_error(self, synthetic_processed_dir):
        dataset = self._dataset(synthetic_processed_dir)
        # Corrupt the first sample's file so loading must fail loudly.
        (dataset.split_dir / dataset.samples[0]["filename"]).unlink()
        with pytest.raises(IOError, match="Failed to load image"):
            dataset[0]


class TestAugmentation:
    """Test data augmentation"""

    # test_augmentation_enabled removed - depends on image files no longer on disk
    
    def test_augmentation_disabled_for_val(self):
        """Test that augmentation is disabled for val even if requested"""
        from pathlib import Path
        import pytest
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        
        manifest = Path("data/processed/utkface") / "val_manifest.json"
        if not manifest.exists():
            pytest.skip(
                "Processed UTKFace val split not available: missing "
                "data/processed/utkface/val_manifest.json. "
                "Run scripts/preprocess_utkface.py first."
            )
        
        dataset = ProcessedUTKFaceDataset(split='val', augment=True)
        
        # Augment should be forced to False for val
        assert dataset.augment == False
        print("✅ Augmentation correctly disabled for validation")


if __name__ == "__main__":
    print("Running UTKFace Dataset Tests...\n")
    print("=" * 60)
    
    try:
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        
        # Test 1: Dataset loads
        dataset = ProcessedUTKFaceDataset(split='train')
        print(f"Test 1 - Train dataset loaded: {len(dataset)} samples")
        
        # Test 2: Statistics
        stats = dataset.get_statistics()
        print(f"Test 2 - Stats: mean age {stats.mean_age:.1f}, "
              f"range {stats.min_age}-{stats.max_age}")
        
        # Test 3: All splits
        val_dataset = ProcessedUTKFaceDataset(split='val')
        test_dataset = ProcessedUTKFaceDataset(split='test')
        print(f"Test 3 - Splits: train={len(dataset)}, val={len(val_dataset)}, "
              f"test={len(test_dataset)}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        raise
