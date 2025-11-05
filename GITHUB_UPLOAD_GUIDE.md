# 📘 PANDUAN LENGKAP: Upload Project ke GitHub

## 🎯 Tujuan
Upload project ChestMNIST Classification ke GitHub repository baru dengan hasil 92.13% accuracy.

---

## 📋 STEP 1: PERSIAPAN PROJECT

### 1.1 Bersihkan File yang Tidak Perlu

Buka PowerShell di folder project:
```powershell
cd D:\vscode\improved-chestmnist\chest-mnist-classification
```

### 1.2 Verifikasi File Penting
Pastikan file-file ini ada:
- ✅ `README_GITHUB.md` (rename nanti jadi README.md)
- ✅ `.gitignore` (sudah dibuat)
- ✅ `requirements.txt`
- ✅ Semua file `.py` (model, training, evaluation)
- ✅ File hasil (`.png`)

### 1.3 Hapus File Temporary (Optional)
```powershell
# Hapus file cache Python
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue

# Hapus model files (terlalu besar untuk GitHub)
# Model files sudah di-ignore di .gitignore
```

---

## 📋 STEP 2: BUAT GITHUB REPOSITORY

### 2.1 Login ke GitHub
1. Buka browser
2. Kunjungi: https://github.com
3. Login dengan akun Anda

### 2.2 Create New Repository
1. Klik tombol **"+" di kanan atas** → **"New repository"**

2. **Isi form:**
   - **Repository name**: `chestmnist-classification`
   - **Description**: `Binary classification of chest X-rays (Cardiomegaly vs Pneumothorax) using DenseNet121 - 92.13% accuracy`
   - **Visibility**: 
     - ✅ **Public** (recommended - bisa di-share)
     - ⬜ Private (jika ingin private)
   
3. **JANGAN centang:**
   - ⬜ Add a README file (kita sudah punya)
   - ⬜ Add .gitignore (kita sudah punya)
   - ⬜ Choose a license (bisa tambah nanti)

4. Klik **"Create repository"**

### 2.3 Catat URL Repository
Setelah dibuat, GitHub akan menampilkan URL seperti:
```
https://github.com/YOUR_USERNAME/chestmnist-classification.git
```
**📝 Simpan URL ini!**

---

## 📋 STEP 3: INISIALISASI GIT LOKAL

### 3.1 Buka PowerShell di Folder Project
```powershell
cd D:\vscode\improved-chestmnist\chest-mnist-classification
```

### 3.2 Cek Apakah Git Sudah Terinstall
```powershell
git --version
```

**Jika error "command not found":**
1. Download Git: https://git-scm.com/download/win
2. Install dengan default settings
3. Restart PowerShell
4. Coba lagi `git --version`

### 3.3 Konfigurasi Git (Jika Belum)
```powershell
git config --global user.name "Nama Anda"
git config --global user.email "email@example.com"
```

**Gunakan email yang sama dengan akun GitHub Anda!**

### 3.4 Inisialisasi Git Repository
```powershell
git init
```

**Output yang diharapkan:**
```
Initialized empty Git repository in D:/vscode/improved-chestmnist/chest-mnist-classification/.git/
```

---

## 📋 STEP 4: RENAME README

Rename file README agar sesuai dengan GitHub:

```powershell
# Backup README lama (jika ada)
if (Test-Path README.md) {
    Move-Item README.md README_old.md -Force
}

# Rename README_GITHUB.md menjadi README.md
Move-Item README_GITHUB.md README.md -Force
```

---

## 📋 STEP 5: TAMBAHKAN FILE KE GIT

### 5.1 Tambahkan Semua File
```powershell
git add .
```

**Ini akan menambahkan semua file KECUALI yang ada di `.gitignore`**

### 5.2 Cek Status
```powershell
git status
```

**Output yang diharapkan:**
```
On branch main

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   .gitignore
        new file:   README.md
        new file:   check_gpu.py
        new file:   datareader.py
        new file:   datareader_highres.py
        new file:   evaluate_densenet.py
        new file:   model.py
        new file:   model_densenet.py
        new file:   model_highres.py
        ... (dan file lainnya)
```

### 5.3 Cek File yang Diabaikan (Optional)
```powershell
git status --ignored
```

**Pastikan file `.pth` dan `__pycache__` ada di "Ignored files"**

---

## 📋 STEP 6: COMMIT PERTAMA

### 6.1 Buat Commit
```powershell
git commit -m "Initial commit: ChestMNIST classification with DenseNet121 (92.13% accuracy)"
```

**Output yang diharapkan:**
```
[main (root-commit) abc1234] Initial commit: ChestMNIST classification with DenseNet121 (92.13% accuracy)
 XX files changed, XXXX insertions(+)
 create mode 100644 .gitignore
 create mode 100644 README.md
 ...
```

---

## 📋 STEP 7: CONNECT KE GITHUB

### 7.1 Tambahkan Remote Repository
```powershell
git remote add origin https://github.com/YOUR_USERNAME/chestmnist-classification.git
```

**⚠️ GANTI `YOUR_USERNAME` dengan username GitHub Anda!**

### 7.2 Verifikasi Remote
```powershell
git remote -v
```

**Output yang diharapkan:**
```
origin  https://github.com/YOUR_USERNAME/chestmnist-classification.git (fetch)
origin  https://github.com/YOUR_USERNAME/chestmnist-classification.git (push)
```

---

## 📋 STEP 8: PUSH KE GITHUB

### 8.1 Rename Branch ke 'main' (Jika Perlu)
```powershell
git branch -M main
```

### 8.2 Push ke GitHub
```powershell
git push -u origin main
```

**Anda akan diminta login:**

**Opsi 1: Browser Authentication (Recommended)**
- Browser akan terbuka
- Login dengan akun GitHub
- Authorize aplikasi
- Kembali ke PowerShell

**Opsi 2: Personal Access Token**
Jika diminta username & password:
1. **Username**: username GitHub Anda
2. **Password**: JANGAN gunakan password asli!
   - Buat Personal Access Token di GitHub:
   - Kunjungi: https://github.com/settings/tokens
   - Klik "Generate new token (classic)"
   - Centang `repo` scope
   - Generate dan copy token
   - Paste sebagai password

**Output yang diharapkan:**
```
Enumerating objects: XX, done.
Counting objects: 100% (XX/XX), done.
Delta compression using up to 8 threads
Compressing objects: 100% (XX/XX), done.
Writing objects: 100% (XX/XX), XXX KiB | XXX MiB/s, done.
Total XX (delta X), reused 0 (delta 0)
To https://github.com/YOUR_USERNAME/chestmnist-classification.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## 📋 STEP 9: VERIFIKASI DI GITHUB

### 9.1 Buka Repository di Browser
```
https://github.com/YOUR_USERNAME/chestmnist-classification
```

### 9.2 Checklist Verifikasi
✅ README.md tampil dengan baik (dengan badges dan formatting)
✅ File-file Python terlihat
✅ Folder structure benar
✅ `.gitignore` berfungsi (tidak ada file `.pth` atau `__pycache__`)
✅ Images (`.png`) muncul di folder results

---

## 📋 STEP 10: TAMBAHAN (OPTIONAL)

### 10.1 Tambahkan License
1. Di GitHub, klik **"Add file"** → **"Create new file"**
2. Nama file: `LICENSE`
3. Klik **"Choose a license template"**
4. Pilih **MIT License**
5. Commit

### 10.2 Buat Folder Results
Jika ingin organize images:

```powershell
# Buat folder results
New-Item -ItemType Directory -Force -Path results

# Pindahkan images
Move-Item *.png results/ -Force

# Add, commit, push
git add results/
git commit -m "Organize result images in results folder"
git push
```

### 10.3 Update README dengan Screenshot
1. Upload images ke GitHub (sudah di folder results)
2. Edit README.md
3. Update path images:
   ```markdown
   ![Training History](results/densenet_training_history.png)
   ```

---

## 📋 STEP 11: UPDATE README DI GITHUB

### 11.1 Edit README Langsung di GitHub
1. Buka repository di browser
2. Klik file `README.md`
3. Klik icon **pensil** (Edit this file)
4. Ganti `YOUR_USERNAME` dengan username GitHub Anda
5. Ganti email dan contact info
6. Klik **"Commit changes"**

**ATAU**

### 11.2 Edit Lokal dan Push
```powershell
# Edit README.md dengan text editor
# Ganti YOUR_USERNAME, email, dll

# Add, commit, push
git add README.md
git commit -m "Update README with correct username and contact info"
git push
```

---

## 📋 TROUBLESHOOTING

### Problem 1: "git: command not found"
**Solusi:**
- Install Git dari https://git-scm.com/download/win
- Restart PowerShell

### Problem 2: "Permission denied (publickey)"
**Solusi:**
- Gunakan HTTPS bukan SSH
- URL: `https://github.com/...` (bukan `git@github.com:...`)

### Problem 3: "Authentication failed"
**Solusi:**
- Buat Personal Access Token di GitHub
- Gunakan token sebagai password

### Problem 4: "File too large" (>100MB)
**Solusi:**
- File `.pth` seharusnya sudah di-ignore
- Jika tetap error:
  ```powershell
  git rm --cached *.pth
  git commit -m "Remove large model files"
  ```

### Problem 5: "Branch 'master' instead of 'main'"
**Solusi:**
```powershell
git branch -M main
git push -u origin main
```

---

## 📋 COMMANDS CHEAT SHEET

```powershell
# Setup
git init
git config --global user.name "Nama"
git config --global user.email "email@example.com"

# Add & Commit
git add .
git commit -m "message"

# Remote
git remote add origin https://github.com/USER/REPO.git
git remote -v

# Push
git branch -M main
git push -u origin main

# Update
git add .
git commit -m "Update message"
git push

# Status
git status
git log --oneline
```

---

## ✅ CHECKLIST FINAL

- [ ] Git installed dan configured
- [ ] GitHub repository created
- [ ] .gitignore sudah benar
- [ ] README.md sudah di-rename
- [ ] Files added dan committed
- [ ] Remote repository connected
- [ ] Pushed to GitHub successfully
- [ ] README tampil dengan baik di GitHub
- [ ] Username dan contact info sudah diupdate
- [ ] License added (optional)
- [ ] Repository public/private sesuai keinginan

---

## 🎉 SELESAI!

Repository Anda sekarang live di:
```
https://github.com/YOUR_USERNAME/chestmnist-classification
```

**Share link ini untuk menunjukkan project Anda!**

---

## 📞 Bantuan Lebih Lanjut

Jika ada masalah, tanyakan kepada saya atau:
- GitHub Docs: https://docs.github.com
- Git Documentation: https://git-scm.com/doc
- Stack Overflow: https://stackoverflow.com/questions/tagged/git

**Good luck! 🚀**
