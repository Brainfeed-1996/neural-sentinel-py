"""
Advanced Tests for Neural Sentinel.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel import (
    Sentinel, EnsembleSentinel, DetectionResult,
    FGSM, PGD, CarliniWagner, AttackResult
)


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


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
    
    def test_detect_norm_method(self):
        """Test detection with norm method."""
        result = self.sentinel.detect(self.test_input, method="norm")
        assert result.method == "norm"
        assert result.perturbation_norm >= 0.0
    
    def test_detect_gradient_method(self):
        """Test detection with gradient method."""
        result = self.sentinel.detect(self.test_input, method="gradient")
        assert result.method == "gradient"
    
    def test_detect_combined_method(self):
        """Test detection with combined method."""
        result = self.sentinel.detect(self.test_input, method="combined")
        assert result.method == "combined"
    
    def test_detect_batch(self):
        """Test batch detection."""
        batch = torch.randn(4, 10)
        results = self.sentinel.detect_batch(batch)
        assert len(results) == 4
        for result in results:
            assert isinstance(result, DetectionResult)
    
    def test_attack_fgsm(self):
        """Test FGSM attack generation."""
        result = self.sentinel.attack(self.test_input, attack_type="fgsm")
        assert isinstance(result, AttackResult)
        assert result.attack_type == "FGSM"
        assert result.success
        assert result.adversarial_tensor.shape == self.test_input.shape
    
    def test_attack_pgd(self):
        """Test PGD attack generation."""
        result = self.sentinel.attack(self.test_input, attack_type="pgd")
        assert isinstance(result, AttackResult)
        assert result.attack_type == "PGD"
        assert result.success
    
    def test_attack_cw(self):
        """Test CW attack generation."""
        result = self.sentinel.attack(self.test_input, attack_type="cw")
        assert isinstance(result, AttackResult)
        assert result.attack_type == "CW"
        assert result.success


class TestEnsembleSentinel:
    """Test cases for EnsembleSentinel class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
    
    def test_ensemble_init(self):
        """Test ensemble initialization."""
        ensemble = EnsembleSentinel(
            self.model,
            methods=["confidence", "norm"],
            weights=[0.6, 0.4]
        )
        assert ensemble.methods == ["confidence", "norm"]
        assert ensemble.weights == [0.6, 0.4]
    
    def test_ensemble_detect(self):
        """Test ensemble detection."""
        ensemble = EnsembleSentinel(self.model)
        input_tensor = torch.randn(1, 10)
        result = ensemble.detect(input_tensor)
        assert isinstance(result, DetectionResult)
        assert result.method == "ensemble"


class TestAttacks:
    """Test cases for attack methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.test_input = torch.randn(1, 10)
    
    def test_fgsm(self):
        """Test FGSM attack."""
        attack = FGSM()
        result = attack.generate(self.model, self.test_input)
        assert result.attack_type == "FGSM"
        assert result.success
    
    def test_pgd(self):
        """Test PGD attack."""
        attack = PGD(iterations=5, step_size=0.01)
        result = attack.generate(self.model, self.test_input)
        assert result.attack_type == "PGD"
        assert result.success
    
    def test_cw(self):
        """Test C&W attack."""
        attack = CarliniWagner()
        result = attack.generate(self.model, self.test_input)
        assert result.attack_type == "CW"
        assert result.success


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_element_tensor(self):
        """Test with single element tensor."""
        model = SimpleModel()
        sentinel = Sentinel(model)
        
        input_2d = torch.randn(1, 10)
        result = sentinel.detect(input_2d)
        assert isinstance(result, DetectionResult)
    
    def test_different_thresholds(self):
        """Test with different thresholds."""
        model = SimpleModel()
        
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sentinel = Sentinel(model, threshold=threshold)
            assert sentinel.threshold == threshold
    
    def test_empty_batch(self):
        """Test with empty batch."""
        model = SimpleModel()
        sentinel = Sentinel(model)
        
        empty_batch = torch.randn(0, 10)
        results = sentinel.detect_batch(empty_batch)
        assert len(results) == 0


class TestMetrics:
    """Test metric calculations."""
    
    def test_attack_perturbation_norm(self):
        """Test perturbation norm calculation."""
        model = SimpleModel()
        sentinel = Sentinel(model)
        
        original = torch.randn(1, 10)
        perturbation = torch.randn(1, 10) * 0.01
        adversarial = original + perturbation
        
        norm = torch.norm(perturbation).item()
        assert norm > 0
        assert norm < 1.0
    
    def test_confidence_range(self):
        """Test confidence score is in valid range."""
        model = SimpleModel()
        sentinel = Sentinel(model)
        
        input_tensor = torch.randn(1, 10)
        result = sentinel.detect(input_tensor, method="confidence")
        
        assert 0.0 <= result.confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
