# Emotion Classification Benchmark

Benchmark framework for facial emotion classification models.

## Structure

```
emotion_detection/
├── benchmark.py          # Main benchmark script
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── core/                 # Core structures and metrics
│   ├── structures.py     # EmotionLabel, BenchmarkMetrics
│   ├── metrics.py        # Accuracy, F1, confusion matrix
│   └── results.py        # Save/load results, HTML reports
├── datasets/             # Dataset loaders
│   ├── base.py           # BaseEmotionDataset
│   ├── fer2013.py        # FER2013 loader
│   ├── affectnet.py      # AffectNet loader
│   └── rafdb.py          # RAF-DB loader
├── classifiers/          # Emotion classifiers
│   ├── base.py           # BaseEmotionClassifier
│   ├── deepface_classifier.py
│   ├── hsemotion.py
│   ├── fer_pytorch.py
│   └── rmn.py
├── runner/               # Benchmark engine
│   └── engine.py
├── scripts/              # Download scripts
│   └── download_datasets.py
├── models/               # Model weights
└── results/              # Benchmark results
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download a dataset

```bash
# FER2013 from Kaggle
python scripts/download_datasets.py --dataset fer2013
```

### 3. Run benchmark

```bash
# All available classifiers on FER2013
python benchmark.py

# Specific classifiers
python benchmark.py --classifiers deepface hsemotion

# With limit and HTML report
python benchmark.py --limit 500 --html --verbose
```

## Supported Classifiers

| Classifier | Model | Install |
|------------|-------|---------|
| DeepFace | CNN (FER2013) | `pip install deepface` |
| HSEmotion | EfficientNet-B0 | `pip install hsemotion` |
| FER | CNN | `pip install fer` |
| RMN | Residual Masking Network | `pip install rmn` |

## Supported Datasets

| Dataset | Classes | Samples | Size |
|---------|---------|---------|------|
| FER2013 | 7 | 35,887 | 48x48 grayscale |
| AffectNet | 8 | ~450,000 | Various |
| RAF-DB | 7 | ~30,000 | Various |

## Emotion Labels

```
0: Angry
1: Disgust
2: Fear
3: Happy
4: Sad
5: Surprise
6: Neutral
7: Contempt (AffectNet only)
```

## Usage Examples

### List available classifiers and datasets

```bash
python benchmark.py --list
```

### Run with specific options

```bash
# Sequential mode (accurate timing)
python benchmark.py --dataset fer2013 --limit 1000

# Parallel mode (faster, but timing not accurate)
python benchmark.py --parallel --workers 4

# Generate HTML report
python benchmark.py --html --verbose

# Save to custom directory
python benchmark.py --output ./my_results
```

### Programmatic usage

```python
from datasets import get_dataset
from classifiers import get_classifier
from runner import run_benchmark

# Load dataset and classifier
dataset = get_dataset("fer2013")
classifier = get_classifier("deepface")

# Run benchmark
metrics = run_benchmark(classifier, dataset, limit=100)

print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"F1 Score: {metrics.macro_f1:.1%}")
print(f"Speed: {metrics.fps:.1f} FPS")
```

## Adding a New Classifier

1. Create `classifiers/my_classifier.py`:

```python
from .base import BaseEmotionClassifier
from core.structures import EmotionLabel

class MyClassifier(BaseEmotionClassifier):
    name = "MyClassifier"
    description = "My custom emotion classifier"

    def _load_model(self):
        # Load your model
        self._model = ...

    def _predict_impl(self, image):
        # Return (label, confidence, probabilities)
        return EmotionLabel.HAPPY, 0.95, {...}
```

2. Register in `classifiers/__init__.py`:

```python
from .my_classifier import MyClassifier
CLASSIFIER_REGISTRY["my_classifier"] = MyClassifier
```

## Output Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1**: Average F1 across all classes
- **Weighted F1**: F1 weighted by class support
- **Per-class metrics**: Precision, Recall, F1 per emotion
- **Confusion matrix**: Detailed error analysis
- **Timing**: ms/image, FPS

## License

MIT License
