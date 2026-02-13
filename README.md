# Neural Sentinel Py

Adversarial example detector for PyTorch models with industrial-grade quality.

## Features

- **Adversarial Detection**: Multiple attack detection methods (FGSM, PGD, CW)
- **Model Agnostic**: Works with any PyTorch model
- **Metrics**: Confidence score, perturbation norm, detection threshold
- **CLI**: Easy integration via command line
- **CI/CD**: Full testing and linting pipeline

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

## CLI

```bash
sentinel --model model.pth --input input.pt --threshold 0.5
```

## Testing

```bash
pytest tests/ -v
```
