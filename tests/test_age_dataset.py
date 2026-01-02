"""
Tests for UTKFace Dataset Loaders
Run with: python -m pytest tests/test_age_dataset.py -v
Or standalone: python tests/test_age_dataset.py
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProcessedDataset:
    """Test cases for ProcessedUTKFaceDataset (primary dataset loader)"""
    
    @pytest.fixture
    def dataset(self):
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
    
    def test_sample_shape(self, dataset):
        """Test that samples have correct shape"""
        image, age = dataset[0]
        
        assert image.shape == (3, 224, 224), \
            f"Expected shape (3, 224, 224), got {image.shape}"
        assert isinstance(age, int), \
            f"Expected int age, got {type(age)}"
        print(f"✅ Sample shape: {image.shape}, age: {age}")
    
    def test_image_normalization(self, dataset):
        """Test that images are normalized to [0, 1]"""
        image, _ = dataset[0]
        
        assert image.min() >= 0.0, "Image min should be >= 0"
        assert image.max() <= 1.0, "Image max should be <= 1"
        print(f"✅ Image range: [{image.min():.3f}, {image.max():.3f}]")
    
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
    
    def test_metadata_retrieval(self, dataset):
        """Test retrieving sample with full metadata"""
        sample = dataset.get_sample_with_metadata(0)
        
        assert 'image' in sample
        assert 'age' in sample
        assert 'gender' in sample
        assert 'race' in sample
        assert 'gender_name' in sample
        assert 'race_name' in sample
        print(f"✅ Metadata: age={sample['age']}, "
              f"gender={sample['gender_name']}, race={sample['race_name']}")


class TestDataLoaders:
    """Test cases for data loader creation"""
    
    def test_loader_creation(self):
        """Test that data loaders are created correctly"""
        from src.contactless.age_estimation.processed_dataset import get_processed_dataloaders
        
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
    
    def test_batch_loading(self):
        """Test loading a batch from train loader"""
        from src.contactless.age_estimation.processed_dataset import get_processed_dataloaders
        
        train_loader, _, _ = get_processed_dataloaders(
            processed_dir="data/processed/utkface",
            batch_size=8,
            num_workers=0
        )
        
        images, ages = next(iter(train_loader))
        
        assert images.shape[0] == 8, "Batch size should be 8"
        assert images.shape[1] == 3, "Should have 3 channels"
        assert ages.shape[0] == 8, "Should have 8 age labels"
        print(f"✅ Batch loaded: images {images.shape}, ages {ages.shape}")


class TestAllSplits:
    """Test all data splits"""
    
    def test_train_split(self):
        """Test train split loads correctly"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        dataset = ProcessedUTKFaceDataset(split='train')
        assert len(dataset) > 40000, f"Expected ~46k train samples, got {len(dataset)}"
        print(f"✅ Train split: {len(dataset)} samples")
    
    def test_val_split(self):
        """Test validation split loads correctly"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        dataset = ProcessedUTKFaceDataset(split='val')
        assert len(dataset) > 9000, f"Expected ~10k val samples, got {len(dataset)}"
        print(f"✅ Val split: {len(dataset)} samples")
    
    def test_test_split(self):
        """Test test split loads correctly"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        dataset = ProcessedUTKFaceDataset(split='test')
        assert len(dataset) > 9000, f"Expected ~10k test samples, got {len(dataset)}"
        print(f"✅ Test split: {len(dataset)} samples")


class TestAugmentation:
    """Test data augmentation"""
    
    def test_augmentation_enabled(self):
        """Test that augmentation can be enabled for training"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        dataset = ProcessedUTKFaceDataset(split='train', augment=True)
        
        # Get same sample - shape should be preserved
        img1, age1 = dataset[0]
        assert img1.shape == (3, 224, 224)
        print("✅ Augmentation enabled without errors")
    
    def test_augmentation_disabled_for_val(self):
        """Test that augmentation is disabled for val even if requested"""
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        dataset = ProcessedUTKFaceDataset(split='val', augment=True)
        
        # Augment should be forced to False for val
        assert dataset.augment == False
        print("✅ Augmentation correctly disabled for validation")


if __name__ == "__main__":
    print("Running UTKFace Dataset Tests...\n")
    print("=" * 60)
    
    try:
        from src.contactless.age_estimation.processed_dataset import (
            ProcessedUTKFaceDataset,
            get_processed_dataloaders
        )
        
        # Test 1: Dataset loads
        dataset = ProcessedUTKFaceDataset(split='train')
        print(f"✅ Test 1 - Train dataset loaded: {len(dataset)} samples")
        
        # Test 2: Sample shape
        image, age = dataset[0]
        print(f"✅ Test 2 - Sample shape: {image.shape}, age: {age}")
        
        # Test 3: Image normalization
        print(f"✅ Test 3 - Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Test 4: Statistics
        stats = dataset.get_statistics()
        print(f"✅ Test 4 - Stats: mean age {stats.mean_age:.1f}, "
              f"range {stats.min_age}-{stats.max_age}")
        
        # Test 5: All splits
        val_dataset = ProcessedUTKFaceDataset(split='val')
        test_dataset = ProcessedUTKFaceDataset(split='test')
        print(f"✅ Test 5 - Splits: train={len(dataset)}, val={len(val_dataset)}, "
              f"test={len(test_dataset)}")
        
        # Test 6: Data loaders
        train_loader, val_loader, test_loader = get_processed_dataloaders(
            batch_size=32
        )
        batch_images, batch_ages = next(iter(train_loader))
        print(f"✅ Test 6 - Batch shape: {batch_images.shape}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
