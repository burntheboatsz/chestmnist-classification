# ====================================================================
# PERMANENT GPU SETUP - ChestMNIST Classification
# ====================================================================

## ✅ STATUS GPU
**ACTIVE** - NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- PyTorch: 2.5.1
- CUDA: 12.1
- Environment: pytorch-gpu (Python 3.11.14)

## 📝 CARA MENGGUNAKAN GPU (3 OPSI)

### 🔥 OPSI 1: Gunakan Script Universal (RECOMMENDED)
```powershell
cd D:\vscode\improved-chestmnist\chest-mnist-classification
.\train_gpu.ps1
```
**Pilih menu:**
1. SimpleCNN Training (Best Model)
2. ResNet18 Training
3. Ensemble Training

**Keuntungan**: Otomatis pakai GPU, ada menu pilihan, error handling

---

### ⚡ OPSI 2: Direct Python dengan GPU Environment
```powershell
# Gunakan full path ke Python GPU
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe train.py

# Atau untuk ensemble
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe train_ensemble.py
```

**Keuntungan**: Langsung, tidak perlu activate environment

---

### 🎯 OPSI 3: Conda Activate (Traditional)
```powershell
# Activate environment
conda activate pytorch-gpu

# Navigate ke project
cd D:\vscode\improved-chestmnist\chest-mnist-classification

# Run training
python train.py
```

**Keuntungan**: Bisa gunakan `python` langsung setelah activate

---

## 🛡️ PROTEKSI ANTI-CPU

Semua script training sudah dimodifikasi untuk **FORCE GPU**:
- `train.py` ✅ Force GPU
- `train_resnet.py` ✅ Force GPU  
- `train_ensemble.py` ✅ Force GPU

**Jika GPU tidak tersedia**, akan muncul error:
```
❌ CUDA/GPU tidak tersedia! Training ini HARUS menggunakan GPU.
```

Ini **SENGAJA** agar tidak accidentally pakai CPU yang lambat!

---

## 📊 CEK GPU STATUS

```powershell
# Quick check
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe check_gpu.py

# Atau
nvidia-smi
```

---

## 🔧 VS CODE SETTINGS (SUDAH DISET)

File `.vscode/settings.json` sudah dikonfigurasi:
```json
{
    "python.defaultInterpreterPath": "C:\\Users\\hnafi\\miniconda3\\envs\\pytorch-gpu\\python.exe"
}
```

**Artinya**: VS Code sudah set default ke pytorch-gpu environment!

---

## ⚙️ ENVIRONMENT DETAILS

**Location**: `C:\Users\hnafi\miniconda3\envs\pytorch-gpu`

**Installed Packages**:
- torch: 2.5.1 (with CUDA 12.1)
- torchvision: 0.20.1
- scikit-learn: 1.7.2
- matplotlib: 3.10.7
- tqdm: 4.67.1
- medmnist: 3.0.2
- pandas, numpy, scipy, dll.

---

## 🚀 QUICK START TRAINING

```powershell
# 1. Check GPU
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe check_gpu.py

# 2. Train SimpleCNN (Best Model)
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe train.py

# 3. Train Ensemble (untuk boost performance)
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe train_ensemble.py
```

---

## 📈 EXPECTED PERFORMANCE WITH GPU

| Model | CPU Time | GPU Time | Validation Acc |
|-------|----------|----------|----------------|
| SimpleCNN | ~45 min | ~5-8 min | 81-84% |
| ResNet18 | ~60 min | ~10-15 min | 80% |
| Ensemble | ~90 min | ~12-18 min | **85-88%** |

**GPU 10x lebih cepat!** 🔥

---

## ⚠️ TROUBLESHOOTING

### Problem: "CUDA not available"
**Solution**: 
```powershell
# Restart PowerShell/VS Code
# Run check
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe check_gpu.py
```

### Problem: "Module not found"
**Solution**:
```powershell
# Install missing package
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\Scripts\pip.exe install <package-name>
```

### Problem: Out of Memory (OOM)
**Solution**: Reduce batch size
```python
# Di train.py atau train_ensemble.py
BATCH_SIZE = 16  # Default 32, turunkan jadi 16
```

---

## 📌 IMPORTANT NOTES

1. ✅ GPU **SELALU** digunakan jika pakai environment `pytorch-gpu`
2. ✅ Semua script training **FORCE GPU** (error jika GPU tidak ada)
3. ✅ VS Code **default interpreter** sudah set ke pytorch-gpu
4. ✅ **Tidak akan berubah-ubah** karena sudah permanent config
5. ✅ Training 10x lebih cepat dengan GPU!

---

## 🎯 RECOMMENDED WORKFLOW

```powershell
# Setiap kali mau training:
cd D:\vscode\improved-chestmnist\chest-mnist-classification
.\train_gpu.ps1  # Menu interaktif

# Atau langsung:
C:\Users\hnafi\miniconda3\envs\pytorch-gpu\python.exe train_ensemble.py
```

**DONE!** GPU sudah menjadi default dan tidak akan berubah! 🚀
