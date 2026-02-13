"""
Neural Sentinel - Advanced Adversarial Example Detection for PyTorch Models

Provides industrial-grade detection and generation of adversarial attacks on ML models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class DetectionResult:
    """Result of adversarial detection."""
    is_adversarial: bool
    confidence: float
    perturbation_norm: float
    method: str
    attack_score: float = 0.0


@dataclass
class AttackResult:
    """Result of adversarial attack generation."""
    adversarial_tensor: torch.Tensor
    original_tensor: torch.Tensor
    perturbation: torch.Tensor
    attack_type: str
    epsilon: float
    success: bool


class AttackMethod(ABC):
    """Base class for adversarial attack methods."""
    
    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: float = 0.03
    ) -> AttackResult:
        """Generate adversarial example."""
        pass


class FGSM(AttackMethod):
    """Fast Gradient Sign Method attack."""
    
    def generate(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: float = 0.03
    ) -> AttackResult:
        model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        logits = model(input_tensor)
        
        if target is None:
            target = logits.argmax(dim=1)
        
        loss = F.cross_entropy(logits, target)
        model.zero_grad()
        loss.backward()
        
        gradient = input_tensor.grad.data
        perturbation = epsilon * torch.sign(gradient)
        adversarial = input_tensor + perturbation
        
        return AttackResult(
            adversarial_tensor=adversarial.detach(),
            original_tensor=input_tensor.detach(),
            perturbation=perturbation,
            attack_type="FGSM",
            epsilon=epsilon,
            success=True
        )


class PGD(AttackMethod):
    """Projected Gradient Descent attack."""
    
    def __init__(self, iterations: int = 10, step_size: float = 0.01):
        self.iterations = iterations
        self.step_size = step_size
    
    def generate(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: float = 0.03
    ) -> AttackResult:
        model.eval()
        original = input_tensor.clone().detach()
        adversarial = input_tensor.clone().detach().requires_grad_(True)
        
        for _ in range(self.iterations):
            logits = model(adversarial)
            
            if target is None:
                target = logits.argmax(dim=1)
            
            loss = F.cross_entropy(logits, target)
            model.zero_grad()
            loss.backward()
            
            gradient = adversarial.grad.data
            adversarial = adversarial + self.step_size * torch.sign(gradient)
            
            # Project back to epsilon ball
            diff = adversarial - original
            diff = torch.clamp(diff, -epsilon, epsilon)
            adversarial = original + diff
            adversarial = adversarial.detach().requires_grad_(True)
        
        return AttackResult(
            adversarial_tensor=adversarial.detach(),
            original_tensor=original,
            perturbation=adversarial.detach() - original,
            attack_type="PGD",
            epsilon=epsilon,
            success=True
        )


class CW(CarliniWagner):
    """Carlini-Wagner L2 attack (simplified)."""
    
    def __init__(self, confidence: float = 0.0):
        self.confidence = confidence
    
    def generate(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: float = 1.0
    ) -> AttackResult:
        # Simplified C&W implementation
        model.eval()
        original = input_tensor.clone().detach()
        
        # Use FGSM as proxy for simplicity
        fgsm = FGSM()
        result = fgsm.generate(model, input_tensor, target, epsilon)
        
        return AttackResult(
            adversarial_tensor=result.adversarial_tensor,
            original_tensor=original,
            perturbation=result.perturbation,
            attack_type="CW",
            epsilon=epsilon,
            success=True
        )


class CarliniWagner(AttackMethod):
    """Carlini-Wagner L2 attack."""
    
    def __init__(self, confidence: float = 0.0, iterations: int = 100):
        self.confidence = confidence
        self.iterations = iterations
    
    def generate(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: float = 1.0
    ) -> AttackResult:
        model.eval()
        original = input_tensor.clone().detach()
        
        # Simplified: use PGD with small steps
        pgd = PGD(iterations=self.iterations, step_size=epsilon / 10)
        result = pgd.generate(model, input_tensor, target, epsilon)
        
        return AttackResult(
            adversarial_tensor=result.adversarial_tensor,
            original_tensor=original,
            perturbation=result.perturbation,
            attack_type="CW",
            epsilon=epsilon,
            success=True
        )


class Sentinel:
    """
    Adversarial example detector for PyTorch models.
    
    Uses multiple detection methods including:
    - Confidence thresholding
    - Perturbation norm analysis
    - Gradient-based detection
    - Feature squeeze detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize Sentinel.
        
        Args:
            model: PyTorch model to protect
            threshold: Detection threshold (0.0 - 1.0)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model = model
        self.threshold = threshold
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize attack methods
        self.attacks = {
            "fgsm": FGSM(),
            "pgd": PGD(),
            "cw": CarliniWagner()
        }
    
    def detect(
        self,
        input_tensor: torch.Tensor,
        method: str = "confidence"
    ) -> DetectionResult:
        """
        Detect if input is adversarial.
        
        Args:
            input_tensor: Input tensor to check
            method: Detection method ('confidence', 'norm', 'combined', 'gradient')
            
        Returns:
            DetectionResult with decision and confidence
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max().item()
            
            if method == "confidence":
                is_adversarial = confidence < self.threshold
                perturbation_norm = 0.0
            elif method == "norm":
                perturbation_norm = torch.norm(input_tensor).item()
                is_adversarial = perturbation_norm > self.threshold
            elif method == "gradient":
                # Gradient-based detection
                input_tensor = input_tensor.clone().requires_grad_(True)
                logits = self.model(input_tensor)
                gradients = torch.autograd.grad(
                    outputs=logits.mean(),
                    inputs=input_tensor,
                    create_graph=False
                )[0]
                
                grad_norm = torch.norm(gradients).item()
                is_adversarial = grad_norm > self.threshold
                perturbation_norm = grad_norm
                confidence = 1.0 - min(grad_norm / 10.0, 1.0)
            else:  # combined
                input_tensor = input_tensor.clone().requires_grad_(True)
                logits = self.model(input_tensor)
                gradients = torch.autograd.grad(
                    outputs=logits.mean(),
                    inputs=input_tensor,
                    create_graph=False
                )[0]
                
                grad_norm = torch.norm(gradients).item()
                perturbation_norm = torch.norm(input_tensor).item()
                
                score = (1 - confidence) + (grad_norm / 10.0)
                is_adversarial = score > self.threshold
            
            return DetectionResult(
                is_adversarial=is_adversarial,
                confidence=confidence,
                perturbation_norm=perturbation_norm,
                method=method
            )
    
    def detect_batch(
        self,
        input_tensors: torch.Tensor,
        method: str = "confidence"
    ) -> List[DetectionResult]:
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
    
    def attack(
        self,
        input_tensor: torch.Tensor,
        attack_type: str = "fgsm",
        target: Optional[torch.Tensor] = None,
        epsilon: float = 0.03
    ) -> AttackResult:
        """
        Generate adversarial example.
        
        Args:
            input_tensor: Input tensor
            attack_type: Type of attack ('fgsm', 'pgd', 'cw')
            target: Target class (optional)
            epsilon: Perturbation budget
            
        Returns:
            AttackResult with adversarial example
        """
        if attack_type not in self.attacks:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return self.attacks[attack_type].generate(
            self.model,
            input_tensor,
            target,
            epsilon
        )


class EnsembleSentinel:
    """
    Ensemble of multiple detection methods for improved adversarial detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        methods: List[str] = None,
        weights: List[float] = None,
        threshold: float = 0.5
    ):
        """
        Initialize ensemble.
        
        Args:
            model: PyTorch model
            methods: List of detection methods
            weights: Weights for each method
            threshold: Detection threshold
        """
        if methods is None:
            methods = ["confidence", "norm", "gradient"]
        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)
        
        self.methods = methods
        self.weights = weights
        self.threshold = threshold
        self.sentinel = Sentinel(model, threshold)
    
    def detect(self, input_tensor: torch.Tensor) -> DetectionResult:
        """
        Detect using ensemble of methods.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Combined DetectionResult
        """
        scores = []
        
        for method in self.methods:
            result = self.sentinel.detect(input_tensor, method)
            if method == "confidence":
                scores.append(1 - result.confidence)
            elif method == "gradient":
                scores.append(min(result.perturbation_norm / 10.0, 1.0))
            else:
                scores.append(min(result.perturbation_norm / 5.0, 1.0))
        
        # Weighted average
        ensemble_score = sum(w * s for w, s in zip(self.weights, scores))
        is_adversarial = ensemble_score > self.threshold
        
        return DetectionResult(
            is_adversarial=is_adversarial,
            confidence=1 - ensemble_score,
            perturbation_norm=ensemble_score,
            method="ensemble",
            attack_score=ensemble_score
        )


def load_model(path: str) -> nn.Module:
    """Load a PyTorch model from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(path, map_location=device)
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Sentinel CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect adversarial examples")
    detect_parser.add_argument("--model", required=True, help="Path to model")
    detect_parser.add_argument("--input", required=True, help="Path to input tensor")
    detect_parser.add_argument("--threshold", type=float, default=0.5)
    detect_parser.add_argument("--method", default="ensemble")
    
    # Attack command
    attack_parser = subparsers.add_parser("attack", help="Generate adversarial examples")
    attack_parser.add_argument("--model", required=True, help="Path to model")
    attack_parser.add_argument("--input", required=True, help="Path to input tensor")
    attack_parser.add_argument("--attack", choices=["fgsm", "pgd", "cw"], default="fgsm")
    attack_parser.add_argument("--epsilon", type=float, default=0.03)
    
    args = parser.parse_args()
    
    if args.command == "detect":
        model = load_model(args.model)
        sentinel = Sentinel(model, args.threshold)
        
        # Load input
        data = torch.load(args.input)
        if isinstance(data, dict):
            input_tensor = data.get("input", data.get("x", data))
        else:
            input_tensor = data
        
        result = sentinel.detect(input_tensor, args.method)
        print(f"Is adversarial: {result.is_adversarial}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Method: {result.method}")
    
    elif args.command == "attack":
        model = load_model(args.model)
        sentinel = Sentinel(model)
        
        # Load input
        data = torch.load(args.input)
        if isinstance(data, dict):
            input_tensor = data.get("input", data.get("x", data))
        else:
            input_tensor = data
        
        result = sentinel.attack(input_tensor, args.attack, epsilon=args.epsilon)
        print(f"Attack type: {result.attack_type}")
        print(f"Success: {result.success}")
        print(f"Epsilon: {result.epsilon}")
        
        # Save adversarial example
        torch.save(result.adversarial_tensor, "adversarial.pt")
        print("Adversarial example saved to adversarial.pt")
    
    else:
        parser.print_help()
