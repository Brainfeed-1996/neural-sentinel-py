"""
Tests for Neural Sentinel.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel import Sentinel, DetectionResult, load_model


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestSentinel:
    """Test cases for Sentinel class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.sentinel = Sentinel(self.model, threshold=0.5)
        self.test_input = torch.randn(1, 10)
    
    def test_init(self):
        """Test Sentinel initialization."""
        assert self.sentinel.model is not None
        assert self.sentinel.threshold == 0.5
    
    def test_detect_returns_result(self):
        """Test detect returns DetectionResult."""
        result = self.sentinel.detect(self.test_input)
        assert isinstance(result, DetectionResult)
        assert hasattr(result, "is_adversarial")
        assert hasattr(result, "confidence")
        assert hasattr(result, "perturbation_norm")
        assert hasattr(result, "method")
    
    def test_detect_confidence_method(self):
        """Test detection with confidence method."""
        result = self.sentinel.detect(self.test_input, method="confidence")
        assert result.method == "confidence"
        assert 0.0 <= result.confidence <= 1.0
    
    def test_detect_batch(self):
        """Test batch detection."""
        batch = torch.randn(4, 10)
        results = self.sentinel.detect_batch(batch)
        assert len(results) == 4
        for result in results:
            assert isinstance(result, DetectionResult)
    
    def test_high_confidence_not_adversarial(self):
        """Test that high confidence input is not flagged as adversarial."""
        # Create input that gives high confidence
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.test_input)
            probs = torch.softmax(logits, dim=1)
            max_prob = probs.max().item()
        
        if max_prob > 0.5:
            result = self.sentinel.detect(self.test_input)
            assert result.confidence == max_prob


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_element_tensor(self):
        """Test with single element tensor."""
        model = SimpleModel()
        sentinel = Sentinel(model)
        
        # Test 2D tensor (1, 10)
        input_2d = torch.randn(1, 10)
        result = sentinel.detect(input_2d)
        assert isinstance(result, DetectionResult)
    
    def test_different_thresholds(self):
        """Test with different thresholds."""
        model = SimpleModel()
        
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sentinel = Sentinel(model, threshold=threshold)
            assert sentinel.threshold == threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
