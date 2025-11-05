# ChestMNIST Binary Classification - DenseNet121

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.13%25-brightgreen.svg)](README.md)

## 🎯 Project Overview

Deep learning project untuk klasifikasi biner penyakit paru-paru menggunakan dataset ChestMNIST (subset dari MedMNIST). Model terbaik mencapai **92.13% validation accuracy** menggunakan DenseNet121 dengan medical pretraining.

### 🏥 Medical Task
- **Binary Classification**: Cardiomegaly vs Pneumothorax
- **Dataset**: ChestMNIST (2,306 training, 305 validation samples)
- **Input**: Grayscale chest X-ray images (128x128)

### 🏆 Best Results

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | **92.13%** |
| **AUC-ROC** | **96.56%** |
| **F1-Score** | **94.34%** |
| **Sensitivity (Recall)** | **96.15%** |
| **Specificity** | **83.51%** |

---

## 📊 Model Performance Comparison

| Model | Resolution | Val Acc | AUC | F1-Score | Parameters |
|-------|-----------|---------|-----|----------|------------|
| SimpleCNN | 28×28 | 81-84% | - | - | 307K |
| SimpleCNN_HighRes | 128×128 | 85.25% | 92.54% | 89.21% | 8.9M |
| Ensemble (SimpleCNN+ResNet) | 28×28 | 83.14% | 89.68% | 87.01% | 11.3M |
| **DenseNet121** ⭐ | **128×128** | **92.13%** | **96.56%** | **94.34%** | **7.6M** |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended: 4GB+ VRAM)
- Conda or virtualenv

### Installation

1. **Clone repository**
```bash
git clone https://github.com/YOUR_USERNAME/chestmnist-classification.git
cd chestmnist-classification
```

2. **Create environment**
```bash
conda create -n pytorch-gpu python=3.11
conda activate pytorch-gpu
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify GPU**
```bash
python check_gpu.py
```

---

## 🎓 Training

### 1. Train DenseNet121 (Best Model - 92.13%)
```bash
python train_densenet.py
```

**Training Configuration:**
- Model: DenseNet121 with ImageNet pretraining
- Input: 128×128 grayscale images
- Batch Size: 16
- Two-Stage Training:
  - Stage 1 (10 epochs): Train classifier head only
  - Stage 2 (50 epochs): Full fine-tuning
- Augmentation: Medical-safe (rotation, affine, sharpness)
- Early Stopping: Patience = 10

**Expected Time:** ~30-45 minutes on RTX 3050

### 2. Train High-Resolution SimpleCNN (85.25%)
```bash
python train_highres.py
```

### 3. Train Ensemble Model (83.14%)
```bash
python train_ensemble.py
```

### 4. Train Baseline SimpleCNN (81-84%)
```bash
python train.py
```

---

## 📈 Evaluation

### Generate Comprehensive Metrics
```bash
python evaluate_densenet.py
```

**Output:**
- Confusion Matrix (raw & normalized)
- ROC Curve (AUC = 96.56%)
- Precision-Recall Curve
- Classification Report
- Error Analysis

**Saved Files:**
- `densenet_evaluation_matrix.png` - 6-panel visualization
- Console output with detailed metrics

### Confusion Matrix Results
```
                Predicted
              Cardio  Pneumo
Actual Cardio    81      16    (83.5% correct)
       Pneumo     8     200    (96.2% correct)
```

**Medical Interpretation:**
- ✅ **High Sensitivity (96.15%)**: Only 8/208 pneumothorax cases missed
- ✅ **Good Specificity (83.51%)**: 81/97 cardiomegaly correctly identified
- ⚠️ **False Negatives: 8** (critical for medical - minimized)
- ⚠️ **False Positives: 16** (acceptable trade-off)

---

## 📁 Project Structure

```
chestmnist-classification/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── check_gpu.py                       # GPU verification script
│
├── datareader.py                      # Baseline data loader (28×28)
├── datareader_highres.py              # High-res data loader (128×128)
│
├── model.py                           # SimpleCNN architecture
├── model_highres.py                   # High-res CNN architectures
├── model_resnet.py                    # ResNet18 transfer learning
├── model_densenet.py                  # DenseNet121 (BEST MODEL) ⭐
├── ensemble_model.py                  # Ensemble architecture
│
├── train.py                           # Train baseline SimpleCNN
├── train_highres.py                   # Train high-res SimpleCNN
├── train_resnet.py                    # Train ResNet18
├── train_densenet.py                  # Train DenseNet121 ⭐
├── train_ensemble.py                  # Train ensemble model
│
├── evaluate_densenet.py               # Comprehensive evaluation
├── utils.py                           # Utility functions
│
├── GUIDE_TO_95_PERCENT.md            # Roadmap to 95% accuracy
├── ROADMAP_TO_95_PERCENT.md          # Alternative roadmap
│
└── results/                           # Training results
    ├── densenet_evaluation_matrix.png
    ├── densenet_training_history.png
    ├── training_history_highres.png
    └── ensemble_training_history.png
```

---

## 🔬 Model Architecture Details

### DenseNet121 (Best Model)

```python
DenseNet121Medical(
  backbone: DenseNet121 (ImageNet pretrained)
  classifier: Sequential(
    Dropout(p=0.5)
    Linear(1024 -> 512)
    ReLU()
    BatchNorm1d(512)
    Dropout(p=0.5)
    Linear(512 -> 256)
    ReLU()
    BatchNorm1d(256)
    Dropout(p=0.25)
    Linear(256 -> 1)  # Binary output
  )
)
```

**Total Parameters:** 7,611,777
- Backbone (frozen Stage 1): 6,953,856
- Classifier (trainable): 657,921

**Key Features:**
- Dense connections for feature reuse
- ImageNet pretrained weights
- Medical-optimized classifier head
- Grayscale to RGB conversion layer
- Strong dropout regularization (0.5)

---

## 🛠️ Technical Details

### Data Preprocessing
1. **Resize**: 28×28 → 128×128 (bicubic interpolation)
2. **Normalization**: [0, 255] → [0, 1]
3. **Augmentation** (medical-safe):
   - Random Rotation: ±5°
   - Random Affine: translate 5%, scale 5%
   - Random Sharpness: 0.5-1.5
   - Auto Contrast

### Training Strategy
**Two-Stage Training:**

**Stage 1 (10 epochs):**
- Freeze DenseNet backbone
- Train classifier head only
- Learning Rate: 0.001
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

**Stage 2 (50 epochs):**
- Unfreeze all layers
- Full fine-tuning
- Learning Rate: 0.0001
- Optimizer: Adam
- Scheduler: CosineAnnealingWarmRestarts
- Early Stopping: Patience = 10

### Loss Function
- **BCEWithLogitsLoss**: Binary Cross-Entropy with Logits
- Combines Sigmoid + BCE for numerical stability

### Regularization
- Dropout: 0.5 (classifier layers)
- Weight Decay: 1e-4
- Gradient Clipping: max_norm = 1.0
- Early Stopping: Patience = 10

---

## 📊 Results & Visualizations

### Training History
![DenseNet Training](results/densenet_training_history.png)

### Evaluation Matrix
![Evaluation Matrix](results/densenet_evaluation_matrix.png)

**Includes:**
1. Confusion Matrix (counts)
2. Normalized Confusion Matrix (%)
3. ROC Curve (AUC = 0.9656)
4. Precision-Recall Curve
5. Metrics Overview
6. Prediction Distribution

---

## 🎯 Roadmap to 95% Accuracy

Current best: **92.13%** | Target: **95%** | Gap: **2.87%**

### Recommended Approaches (Ranked by Success Probability)

#### 1. Test-Time Augmentation (TTA) - Expected: +1-2%
- Augment images during inference
- Average predictions from multiple versions
- Implementation time: 1-2 hours

#### 2. Ensemble DenseNet + SimpleCNN_HighRes - Expected: +1-2%
- Combine DenseNet (92.13%) + SimpleCNN_HighRes (85.25%)
- Weighted averaging or learnable weights
- Implementation time: 2-3 hours

#### 3. K-Fold Cross-Validation - Expected: +2-3%
- Train 5 models with different data splits
- Ensemble all folds
- Implementation time: 5-10 hours

#### 4. Advanced Augmentation - Expected: +0.5-1%
- Mixup / CutMix
- GridMask
- Implementation time: 2-4 hours

#### 5. External Pretraining - Expected: +1-3%
- Use CheXNet weights (NIH ChestX-ray14 pretrained)
- torchxrayvision library
- Implementation time: 1 week

**See `GUIDE_TO_95_PERCENT.md` for detailed implementation guide.**

---

## 💻 Hardware Requirements

### Minimum
- GPU: 4GB VRAM (e.g., RTX 3050)
- RAM: 8GB
- Storage: 5GB

### Recommended
- GPU: 6GB+ VRAM (e.g., RTX 3060)
- RAM: 16GB
- Storage: 10GB

### Tested On
- GPU: NVIDIA GeForce RTX 3050 Laptop (4GB)
- CUDA: 12.1
- Driver: 581.57
- PyTorch: 2.5.1+cu121

---

## 📦 Dependencies

**Core:**
- PyTorch 2.5.1
- torchvision 0.20.1
- medmnist 3.0.2
- scikit-learn 1.6.1
- numpy 2.3.1

**Visualization:**
- matplotlib 3.10.7
- seaborn 0.13.2

**See `requirements.txt` for complete list.**

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
1. Implement Test-Time Augmentation
2. Add K-Fold Cross-Validation
3. Experiment with CheXNet pretrained weights
4. Optimize hyperparameters
5. Add more medical-specific augmentations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MedMNIST**: Yang, J., et al. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." (2023)
- **DenseNet**: Huang, G., et al. "Densely Connected Convolutional Networks." CVPR 2017
- **ChestX-ray14**: Wang, X., et al. "ChestX-ray8: Hospital-scale Chest X-ray Database." CVPR 2017

---

## 📧 Contact

For questions or collaborations:
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com

---

## 📈 Citation

If you use this code in your research, please cite:

```bibtex
@software{chestmnist_densenet,
  author = {Your Name},
  title = {ChestMNIST Binary Classification with DenseNet121},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/chestmnist-classification}
}
```

---

**⭐ Star this repo if you find it helpful!**

**Target Achieved: 92.13% Validation Accuracy** 🎉
