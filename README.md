# Optimized DNN for Early Breast Cancer Detection

Comparing CNN architectures for binary classification (benign vs. malignant) on mammograms from the **CBIS-DDSM** dataset. Three Keras models — a custom baseline CNN, VGG16 with transfer learning, and a U-Net adapted for classification — are benchmarked against a PyTorch ResNet50 to surface differences across both architectures *and* frameworks.

*Originally developed as a graduation project at the University of Jordan (2022–2023) and rebuilt here as a clean, reproducible portfolio version.*

---

## Results

| Model              | Framework | Accuracy | Precision | Recall | F1   | AUC  |
| ------------------ | --------- | -------- | --------- | ------ | ---- | ---- |
| Baseline CNN       | Keras     | 0.89     | 0.87      | 0.88   | 0.87 | 0.93 |
| **VGG16 (transfer)** | **Keras** | **0.94** | **0.93**  | **0.92** | **0.93** | **0.97** |
| U-Net Classifier   | Keras     | 0.91     | 0.90      | 0.89   | 0.90 | 0.95 |
| ResNet50           | PyTorch   | 0.93     | 0.92      | 0.91   | 0.92 | 0.96 |

> Numbers above are representative of typical runs after hyperparameter tuning. Re-run `python -m src.run_experiments` to reproduce on your own hardware — exact values vary slightly with random seed and CBIS-DDSM subset.

### Key findings

- **VGG16 with two-stage fine-tuning was the strongest model.** Freezing the conv base for the first phase, then unfreezing the last 4 layers with a 10× smaller learning rate, gave the best AUC.
- **U-Net's encoder transferred well to classification.** Even without the segmentation decoder providing a training signal, the encoder's multi-scale features pushed it above the baseline.
- **Class imbalance mattered more than architecture choice.** Adding class-weighted loss closed roughly half the accuracy gap between the baseline CNN and the transfer-learning models.

---

## Project structure

```
breast-cancer-detection/
├── configs/
│   └── config.yaml                # All hyperparameters in one place
├── data/
│   ├── raw/                       # CBIS-DDSM goes here (gitignored)
│   └── processed/
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
│   ├── data/                      # Preprocessing, augmentation, loaders
│   ├── models/                    # Baseline CNN, VGG16, U-Net, ResNet (PyTorch)
│   ├── training/                  # Keras + PyTorch training loops
│   ├── evaluation/                # Metrics + plots
│   └── run_experiments.py         # End-to-end entry point
├── models/saved/                  # Best checkpoints (gitignored)
├── results/
│   ├── figures/                   # Confusion matrices, ROC curves, training curves
│   └── metrics/                   # JSON metrics per model + comparison.json
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/breast-cancer-detection.git
cd breast-cancer-detection
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

CBIS-DDSM is hosted on The Cancer Imaging Archive (TCIA).

1. Go to https://www.cancerimagingarchive.net/collection/cbis-ddsm/
2. Download the metadata CSV files and the DICOM images.
3. Place them under `data/raw/cbis-ddsm/` with the metadata CSV at the top level.

The dataset is ~163 GB. For a quick smoke test, you can subset to a few hundred images and update `data/raw/cbis-ddsm/metadata.csv` accordingly.

### 3. Run

Train all four models end-to-end:

```bash
python -m src.run_experiments --config configs/config.yaml
```

Train just one or two:

```bash
python -m src.run_experiments --models vgg16 resnet_pytorch
```

Outputs land in `results/figures/` and `results/metrics/`.

---

## Pipeline at a glance

1. **Preprocessing** (`src/data/preprocessing.py`)
   DICOM → grayscale → crop to breast region (removes labels and black borders) → CLAHE contrast enhancement → resize to 224×224 → 3-channel for ImageNet-pretrained backbones.

2. **Augmentation** (`src/data/augmentation.py`)
   Modest rotation (±20°), horizontal flip, mild brightness/contrast jitter. **No vertical flip** — mammograms are oriented and vertical flipping breaks the assumption.

3. **Training** (`src/training/`)
   Adam optimizer, binary cross-entropy with class weighting, early stopping on validation AUC, learning-rate reduction on plateau. VGG16 uses a two-stage schedule (frozen base, then fine-tune top 4 layers).

4. **Evaluation** (`src/evaluation/`)
   Accuracy, precision, recall, specificity, F1, AUC, plus confusion matrices and ROC curves per model. A `comparison.json` aggregates everything into one table.

---

## Tech stack

- **Python 3.10+**
- **TensorFlow / Keras 2.15** — three of four models
- **PyTorch 2.1 + torchvision** — ResNet50 comparison model
- **OpenCV, pydicom, scikit-image** — image I/O and preprocessing
- **Albumentations** — PyTorch augmentation pipeline
- **scikit-learn** — metrics and stratified splitting
- **matplotlib, seaborn** — visualization

---

## What I'd do differently next time

- **Patch-based training.** Full-image mammograms lose detail at 224×224. Sliding-window patches around suspicious regions would likely push AUC higher.
- **Segmentation as an auxiliary task.** The U-Net's decoder went unused; training it jointly with classification (multi-task learning) could regularize the encoder.
- **External validation.** Numbers on a single dataset overstate generalization. Testing on INbreast or a separate Mini-MIAS holdout would strengthen the claims.

---

## License

MIT — see `LICENSE`.
