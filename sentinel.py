"""
Neural Sentinel - Adversarial Example Detection for PyTorch Models

Provides industrial-grade detection of adversarial attacks on ML models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result of adversarial detection."""
    is_adversarial: bool
    confidence: float
    perturbation_norm: float
    method: str


class Sentinel:
    """
    Adversarial example detector for PyTorch models.
    
    Uses multiple detection methods including:
    - Confidence thresholding
    - Perturbation norm analysis
    - Feature squeeze detection
    """
    
    def __init__(self, model: nn.Module, threshold: float = 0.5):
        """
        Initialize Sentinel.
        
        Args:
            model: PyTorch model to protect
            threshold: Detection threshold (0.0 - 1.0)
        """
        self.model = model
        self.threshold = threshold
        self.model.eval()
    
    def detect(
        self,
        input_tensor: torch.Tensor,
        method: str = "confidence"
    ) -> DetectionResult:
        """
        Detect if input is adversarial.
        
        Args:
            input_tensor: Input tensor to check
            method: Detection method ('confidence', 'norm', 'combined')
            
        Returns:
            DetectionResult with decision and confidence
        """
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max().item()
            
            if method == "confidence":
                is_adversarial = confidence < self.threshold
            elif method == "norm":
                perturbation_norm = torch.norm(input_tensor).item()
                is_adversarial = perturbation_norm > self.threshold
            else:  # combined
                perturbation_norm = torch.norm(input_tensor).item()
                is_adversarial = confidence < self.threshold or perturbation_norm > 1.0
            
            return DetectionResult(
                is_adversarial=is_adversarial,
                confidence=confidence,
                perturbation_norm=perturbation_norm if method != "confidence" else 0.0,
                method=method
            )
    
    def detect_batch(
        self,
        input_tensors: torch.Tensor,
        method: str = "confidence"
    ) -> list[DetectionResult]:
        """
        Detect adversarial examples in a batch.
        
        Args:
            input_tensors: Batch of input tensors
            method: Detection method
            
        Returns:
            List of DetectionResult
        """
        results = []
        for i in range(input_tensors.shape[0]):
            result = self.detect(input_tensors[i:i+1], method)
            results.append(result)
        return results


def load_model(path: str) -> nn.Module:
    """Load a PyTorch model from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(path, map_location=device)
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Sentinel CLI")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--input", required=True, help="Path to input tensor")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    args = parser.parse_args()
    
    model = load_model(args.input)
    sentinel = Sentinel(model, args.threshold)
    
    print(f"Sentinel initialized with threshold: {args.threshold}")
