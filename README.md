# Neural Sentinel Py

Advanced adversarial example detector for PyTorch models with industrial-grade quality.

## Features

- **Adversarial Detection**: Multiple attack detection methods (FGSM, PGD, CW, DeepFool)
- **Ensemble Methods**: Combine multiple detection strategies
- **Model Agnostic**: Works with any PyTorch model
- **Metrics**: Confidence score, perturbation norm, detection threshold, ROC-AUC
- **CLI**: Easy integration via command line
- **CI/CD**: Full testing, linting, and coverage pipeline

## Installation

```bash
pip install -e .
```

## Usage

```python
from sentinel import Sentinel, load_model

model = load_model("model.pth")
sentinel = Sentinel(model)

is_adversarial, confidence = sentinel.detect(input_tensor)
```

## Advanced Usage

```python
from sentinel import EnsembleSentinel

# Combine multiple detection methods
ensemble = EnsembleSentinel(
    model,
    methods=["confidence", "norm", "gradients"],
    weights=[0.4, 0.3, 0.3]
)

result = ensemble.detect(input_tensor)
print(f"Is adversarial: {result.is_adversarial}")
print(f"Confidence: {result.confidence:.4f}")
```

## CLI

```bash
# Basic detection
sentinel --model model.pth --input input.pt --threshold 0.5

# Ensemble mode
sentinel --model model.pth --input input.pt --ensemble --methods confidence norm gradients

# Generate attack
sentinel --model model.pth --attack fgsm --eps 0.03 --output attack.pt
```

## Supported Attacks

| Attack | Method | Description |
|--------|--------|-------------|
| FGSM | `fgsm` | Fast Gradient Sign Method |
| PGD | `pgd` | Projected Gradient Descent |
| CW | `cw` | Carlini-Wagner |
| DeepFool | `deepfool` | Iterative attack |

## Testing

```bash
pytest tests/ -v --cov=sentinel --cov-report=term
```

## License

MIT
