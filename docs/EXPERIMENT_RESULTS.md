# Hasil Eksperimen Model - ChestMNIST Classification

## 📊 Perbandingan Semua Model

| Run | Model | Strategy | Train Acc | Val Acc | Train-Val Gap | Notes |
|-----|-------|----------|-----------|---------|---------------|-------|
| #1 | SimpleCNN | Baseline (2 conv layers) | 84.62% | 78.15% | 6.47% | Original model |
| #2 | SimpleCNN | ✅ **3 conv + BatchNorm + Dropout** | 85.86% | **83.58%** | 2.28% | **BEST SimpleCNN** |
| #3 | SimpleCNN | + Data Augmentation | 72.90% | 79.62% | -6.72% | Augmentation terlalu agresif ❌ |
| #4 | SimpleCNN | + Weighted Loss | 81.18% | 81.09% | 0.09% | Tidak improve ❌ |
| #5 | ResNet18 | Transfer Learning (LR=0.0001) | 91.76% | 71.70% | 20.06% | **Severe overfitting** ❌ |
| #6 | ResNet18 | **Fine-tuned (LR=0.001, Dropout=0.5)** | TBD | TBD | TBD | **RUNNING...** ⏳ |

---

## 🏆 Current Best Model

**SimpleCNN Enhanced (Run #2)**
- **Validation Accuracy**: 83.58%
- **Architecture**: 3 conv layers + BatchNorm + Dropout + MaxPool
- **Regularization**: Dropout2d(0.25), Dropout(0.3), Weight Decay(1e-4)
- **Training**: Early stopping, ReduceLROnPlateau scheduler
- **Device**: CUDA (NVIDIA RTX 3050)

---

## 📈 Lessons Learned

### ✅ What Worked
1. **Model Architecture Improvements**
   - Adding 3rd convolutional layer: +5.43% accuracy
   - BatchNorm for training stability
   - Dropout for preventing overfitting
   - MaxPool instead of AvgPool for better feature selection

2. **Training Techniques**
   - Early stopping (patience=7) - stopped overfitting
   - ReduceLROnPlateau scheduler - adaptive learning rate
   - GPU acceleration - faster training

3. **Regularization**
   - Dropout2d(0.25) after conv blocks
   - Dropout(0.3) in FC layers
   - Weight decay (L2 regularization)

### ❌ What Didn't Work
1. **Data Augmentation** (-3.96% accuracy)
   - RandomRotation, RandomAffine terlalu agresif
   - Medical images sensitif - augmentasi menghilangkan detail diagnostik
   - ColorJitter tidak cocok untuk chest X-ray

2. **Weighted Loss** (-2.49% accuracy)
   - Class imbalance (1:2) tidak signifikan untuk weighted loss
   - Standard BCE loss sudah cukup baik

3. **ResNet18 dengan LR Rendah** (-11.88% accuracy)
   - LR=0.0001 terlalu kecil - model tidak converge dengan baik
   - Overfitting parah (train-val gap 20%)
   - Transfer learning perlu hyperparameter tuning yang tepat

---

## 🔬 ResNet18 Fine-tuning Experiment (Run #6)

### Modifications from Run #5:
- **Learning Rate**: 0.0001 → **0.001** (10x increase)
- **Dropout**: 0.3 → **0.5** (stronger regularization)
- **Weight Decay**: 1e-4 → **5e-4** (5x stronger L2 reg)
- **Epochs**: 25 → **30**
- **Patience**: 5 → **7** (more time to converge)

### Expected Improvements:
1. Higher LR → Better convergence
2. Higher Dropout → Reduce overfitting (target gap < 5%)
3. Stronger Weight Decay → Better generalization
4. More patience → Find better local minimum

### Target:
- **Validation Accuracy**: 85-90%
- **Train-Val Gap**: < 5%
- **Training Time**: ~10-15 minutes

---

## 📝 Next Steps

### If ResNet18 (Run #6) achieves 85%+:
✅ **Success!** Use ResNet18 as final model
- Save model weights
- Document architecture
- Create inference script

### If ResNet18 (Run #6) still underperforms:
1. **Try DenseNet121** - Used by CheXNet (Stanford)
2. **Try EfficientNet-B0** - Best efficiency/accuracy ratio
3. **Ensemble Methods** - Combine SimpleCNN + ResNet18

### If all transfer learning fails:
✅ **Use SimpleCNN Enhanced (Run #2)** - Already excellent at 83.58%
- Proven to work well
- No overfitting issues
- Fast training and inference

---

## 🎯 Conclusion (So Far)

**SimpleCNN Enhanced** is currently the most reliable model with:
- ✅ High accuracy (83.58%)
- ✅ Low overfitting (train-val gap 2.28%)
- ✅ Fast training (~5 minutes)
- ✅ Stable and reproducible

Transfer learning with ResNet18 is being optimized. If successful, it could provide +2-5% improvement.

**Final decision**: Wait for Run #6 results...

---

*Last Updated: Training Run #6 in progress...*
