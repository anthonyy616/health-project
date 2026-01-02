"""
Tests for Age Estimation Model and Training
Run with: python tests/test_age_model.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestModelArchitecture:
    """Test cases for LightAgeNet model"""
    
    def test_light_age_net_forward(self):
        """Test LightAgeNet forward pass"""
        from models.age_detection.light_age_net import LightAgeNet
        
        model = LightAgeNet()
        x = torch.randn(2, 3, 224, 224)  # Batch of 2
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2,), f"Expected output shape (2,), got {y.shape}"
        print(f"✅ LightAgeNet forward pass: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    
    def test_light_age_net_v2_forward(self):
        """Test LightAgeNetV2 forward pass"""
        from models.age_detection.light_age_net import LightAgeNetV2
        
        model = LightAgeNetV2()
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2,), f"Expected output shape (2,), got {y.shape}"
        print(f"✅ LightAgeNetV2 forward pass: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    
    def test_parameter_count(self):
        """Test parameter counts are reasonable"""
        from models.age_detection.light_age_net import LightAgeNet, LightAgeNetV2
        
        light = LightAgeNet()
        v2 = LightAgeNetV2()
        
        light_params = light.get_num_parameters()
        v2_params = v2.get_num_parameters()
        
        # LightAgeNet ~1.6M params (still much smaller than MobileNetV3's 5M+)
        assert 1_000_000 < light_params < 3_000_000, \
            f"LightAgeNet should have ~1.6M params, got {light_params:,}"
        # LightAgeNetV2 ~4.8M params  
        assert 3_000_000 < v2_params < 6_000_000, \
            f"LightAgeNetV2 should have ~4.8M params, got {v2_params:,}"
        
        print(f"✅ Parameter counts: LightAgeNet={light_params:,}, V2={v2_params:,}")
    
    def test_create_model_factory(self):
        """Test model factory function"""
        from models.age_detection.light_age_net import create_model
        
        model_light = create_model("light")
        model_v2 = create_model("v2")
        
        assert model_light.__class__.__name__ == "LightAgeNet"
        assert model_v2.__class__.__name__ == "LightAgeNetV2"
        
        print("✅ Model factory creates correct model types")


class TestEstimator:
    """Test cases for AgeEstimator inference wrapper"""
    
    def test_estimator_initialization(self):
        """Test estimator can be created (without trained weights)"""
        from src.contactless.age_estimation.estimator import AgeEstimator
        
        # Will warn about missing weights but should work
        estimator = AgeEstimator(model_path=None)
        
        # Model should be loaded (random weights)
        assert estimator.model is not None
        print("✅ AgeEstimator initializes correctly")
    
    def test_estimator_preprocess(self):
        """Test image preprocessing"""
        from src.contactless.age_estimation.estimator import AgeEstimator
        
        estimator = AgeEstimator(model_path=None)
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        tensor = estimator.preprocess(dummy_image)
        
        assert tensor.shape == (1, 3, 224, 224), \
            f"Expected shape (1, 3, 224, 224), got {tensor.shape}"
        assert tensor.min() >= 0 and tensor.max() <= 1, \
            "Tensor should be normalized to [0, 1]"
        
        print(f"✅ Preprocessing: {dummy_image.shape} -> {tuple(tensor.shape)}")
    
    def test_estimator_inference(self):
        """Test full inference pipeline"""
        from src.contactless.age_estimation.estimator import AgeEstimator
        
        estimator = AgeEstimator(model_path=None)
        
        # Create dummy face image
        dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        result = estimator.estimate(dummy_face)
        
        assert 0 <= result.age <= 100, f"Age {result.age} out of valid range"
        assert 0 <= result.confidence <= 1, f"Confidence {result.confidence} out of range"
        assert result.inference_time_ms > 0, "Inference time should be positive"
        
        print(f"✅ Inference: age={result.age}, confidence={result.confidence:.2f}, "
              f"time={result.inference_time_ms:.1f}ms")
    
    def test_estimator_callable(self):
        """Test estimator can be called directly"""
        from src.contactless.age_estimation.estimator import AgeEstimator
        
        estimator = AgeEstimator(model_path=None)
        dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        age, confidence = estimator(dummy_face)
        
        assert isinstance(age, int)
        assert isinstance(confidence, float)
        
        print(f"✅ Callable interface works: age={age}, confidence={confidence:.2f}")


class TestIntegration:
    """Integration tests with dataset"""
    
    def test_model_with_real_data(self):
        """Test model with real preprocessed data"""
        from models.age_detection.light_age_net import LightAgeNet
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        
        # Load a sample from dataset
        try:
            dataset = ProcessedUTKFaceDataset(split='test')
        except FileNotFoundError:
            print("⚠ Skipping: Processed dataset not found")
            return
        
        image, true_age = dataset[0]
        
        # Run through model
        model = LightAgeNet()
        model.eval()
        
        with torch.no_grad():
            pred_age = model(image.unsqueeze(0)).item()
        
        print(f"✅ Model with real data: true_age={true_age}, pred_age={pred_age:.1f} "
              f"(error={abs(pred_age - true_age):.1f})")


def run_all_tests():
    """Run all test classes"""
    print("=" * 60)
    print("AGE ESTIMATION MODEL TESTS")
    print("=" * 60)
    
    # Model tests
    print("\n📦 Model Architecture Tests:")
    model_tests = TestModelArchitecture()
    model_tests.test_light_age_net_forward()
    model_tests.test_light_age_net_v2_forward()
    model_tests.test_parameter_count()
    model_tests.test_create_model_factory()
    
    # Estimator tests
    print("\n🔧 Estimator Tests:")
    estimator_tests = TestEstimator()
    estimator_tests.test_estimator_initialization()
    estimator_tests.test_estimator_preprocess()
    estimator_tests.test_estimator_inference()
    estimator_tests.test_estimator_callable()
    
    # Integration tests
    print("\n🔗 Integration Tests:")
    integration_tests = TestIntegration()
    integration_tests.test_model_with_real_data()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
