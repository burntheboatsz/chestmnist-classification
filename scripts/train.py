# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import SimpleCNN
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 30  # Lebih banyak epoch dengan early stopping
BATCH_SIZE = 32  # Batch size lebih besar untuk stabilitas
LEARNING_RATE = 0.001  # Learning rate lebih besar di awal
DROPOUT_RATE = 0.3  # Dropout untuk regularisasi
EARLY_STOP_PATIENCE = 7  # Stop jika tidak ada improvement dalam 7 epoch
USE_DATA_AUGMENTATION = False  # BEST: Tanpa augmentation (augmentation menurunkan performa)
USE_WEIGHTED_LOSS = False  # BEST: Tanpa weighted loss (tidak efektif untuk dataset ini)

#Menampilkan plot riwayat training dan validasi setelah training selesai.

def train():
    # Force GPU Usage - Raise error jika GPU tidak tersedia
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ CUDA/GPU tidak tersedia! Training ini HARUS menggunakan GPU.\n"
            "Pastikan PyTorch dengan CUDA sudah terinstall dengan benar.\n"
            "Gunakan environment: conda activate pytorch-gpu"
        )
    
    device = torch.device('cuda')
    print(f"✅ Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE, use_augmentation=USE_DATA_AUGMENTATION)
    
    # 2. Inisialisasi Model dengan Dropout (BEST: 28x28 input)
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes, dropout_rate=DROPOUT_RATE, input_size=28).to(device)
    print(model)
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    # Gunakan BCEWithLogitsLoss untuk klasifikasi biner (tanpa weighted loss)
    criterion = nn.BCEWithLogitsLoss()
    print("✅ Menggunakan Standard BCE Loss")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # L2 regularization
    
    # Learning Rate Scheduler - mengurangi LR saat plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Early Stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n--- Memulai Training ---")
    print(f"Hyperparameters:")
    print(f"  - Epochs: {EPOCHS} (max)")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Dropout: {DROPOUT_RATE}")
    print(f"  - Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"  - Input Resolution: 28x28 (BEST)")
    print(f"  - Data Augmentation: {'AKTIF' if USE_DATA_AUGMENTATION else 'TIDAK AKTIF (BEST)'}")
    print()
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            # Ubah tipe data label menjadi float untuk BCEWithLogitsLoss
            labels = labels.float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels) # Loss dihitung antara output tunggal dan label
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update Learning Rate Scheduler
        scheduler.step(avg_val_loss)
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% ⭐ NEW BEST!")
        else:
            patience_counter += 1
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% "
                  f"[Patience: {patience_counter}/{EARLY_STOP_PATIENCE}]")
        
        # Check Early Stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early Stopping triggered at epoch {epoch+1}")
            print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model with Val Acc: {best_val_acc:.2f}%")

    print("--- Training Selesai ---")
    
    # Simpan model terbaik
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
    }, 'best_model.pth')
    print("✅ Model terbaik disimpan sebagai 'best_model.pth'")
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()
    