# Ensemble Model - SimpleCNN + ResNet18

## 🎯 Tujuan
Menggabungkan dua model terbaik untuk meningkatkan performa dari **81-84%** menjadi **85-88%**:
1. **SimpleCNN Enhanced** (model terbaik, 81-84% akurasi)
2. **ResNet18** (model transfer learning, 80% akurasi)

## 🔧 Setup Environment dengan GPU

### Masalah: PyTorch CPU-only
PyTorch yang terinstall adalah versi CPU (`2.9.0+cpu`) karena Python 3.13 terlalu baru.

### Solusi: Buat Conda Environment Baru
```powershell
# 1. Create environment dengan Python 3.11
conda create -n pytorch-gpu python=3.11 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 2. Activate environment
conda activate pytorch-gpu

# 3. Install dependencies lain
conda install scikit-learn matplotlib tqdm medmnist albumentations -y
```

## 🚀 Cara Menjalankan

### Option 1: PowerShell Script (Recommended)
```powershell
cd D:\vscode\improved-chestmnist\chest-mnist-classification
.\run_ensemble_gpu.ps1
```

### Option 2: Manual dengan Conda
```powershell
conda activate pytorch-gpu
cd D:\vscode\improved-chestmnist\chest-mnist-classification
python train_ensemble.py
```

## 📊 Strategi Ensemble

### Stage 1: Train Ensemble Weights Only (10 epochs)
- Freeze kedua base model (SimpleCNN + ResNet18)
- Train hanya learnable weights untuk averaging
- Learning rate: 0.01 (higher untuk weights saja)
- Output: `best_ensemble_weights.pth`

### Stage 2: Full Fine-tuning (30-50 epochs)
- Unfreeze semua parameters
- Fine-tune entire ensemble end-to-end
- Learning rate: 0.0001 (lower untuk fine-tuning)
- Early stopping patience: 10
- Output: `best_ensemble_model.pth`

## 🧠 Arsitektur Ensemble

```python
class EnsembleModel:
    - SimpleCNN Enhanced (81-84%)
    - ResNet18 (80%)
    - Learnable weights: [0.6, 0.4]  # SimpleCNN lebih baik
    
    Output = softmax(weights)[0] * SimpleCNN + softmax(weights)[1] * ResNet18
```

## 📈 Expected Performance

| Method | Validation Accuracy |
|--------|-------------------|
| SimpleCNN Enhanced | 81-84% |
| ResNet18 | 80% |
| **Ensemble (Weighted Avg)** | **85-88%** ✨ |

## 📁 Output Files

1. `best_ensemble_weights.pth` - Stage 1 (weights only)
2. `best_ensemble_model.pth` - Stage 2 (full fine-tuned)
3. `ensemble_training_history.png` - Training curves
4. Console output dengan metrics lengkap

## ⚡ GPU Requirements

- **GPU**: NVIDIA RTX 3050 Laptop (4GB VRAM)
- **CUDA**: 12.1 atau 13.0
- **Driver**: NVIDIA 581.57+
- **Memory**: ~2GB VRAM untuk batch_size=32

## 🔍 Verification

Cek CUDA availability:
```python
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA Version: {torch.version.cuda}')
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
CUDA Version: 12.1
```

## 🎓 Advanced: Voting Ensemble

Jika weighted averaging kurang bagus, gunakan `VotingEnsemble` atau `AveragingEnsemble`:

```python
from ensemble_model import VotingEnsemble, AveragingEnsemble

# Hard voting
voting = VotingEnsemble([model1, model2], device='cuda')
pred = voting.predict(images)

# Soft voting dengan custom weights
averaging = AveragingEnsemble([model1, model2], weights=[0.7, 0.3])
pred = averaging.predict(images)
```

## ⏱️ Training Time Estimate

- **CPU**: ~45-60 minutes (sangat lambat)
- **GPU (RTX 3050)**: ~10-15 minutes ⚡

## 📌 Notes

- Pastikan `best_model.pth` (SimpleCNN) dan `best_model_resnet.pth` (ResNet18) sudah ada
- Ensemble menggunakan 28x28 resolution (sesuai SimpleCNN terbaik)
- Data augmentation: DISABLED (terbukti harmful)
- Batch size: 32 (optimal untuk RTX 3050 4GB VRAM)
