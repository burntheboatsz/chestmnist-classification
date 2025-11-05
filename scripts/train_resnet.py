# train_resnet.py
# Training script untuk ResNet18 Transfer Learning

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders
from model_resnet import get_model
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
MODEL_NAME = 'resnet18'  # 'resnet18', 'densenet121', atau 'efficientnet_b0'
EPOCHS = 30  # Lebih banyak epoch untuk convergence yang lebih baik
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # LR lebih tinggi untuk better convergence
DROPOUT_RATE = 0.5  # Dropout lebih tinggi untuk mengurangi overfitting
EARLY_STOP_PATIENCE = 7  # Patience lebih tinggi
USE_PRETRAINED = True  # PENTING: Gunakan pretrained weights
FREEZE_BACKBONE = False  # False = fine-tune all layers
WEIGHT_DECAY = 5e-4  # L2 regularization lebih kuat

def train():
    # Force GPU Usage - Raise error jika GPU tidak tersedia
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ CUDA/GPU tidak tersedia! Training ResNet HARUS menggunakan GPU.\n"
            "Pastikan PyTorch dengan CUDA sudah terinstall.\n"
            "Gunakan environment: conda activate pytorch-gpu"
        )
    
    device = torch.device('cuda')
    print(f"✅ Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    
    # 1. Memuat Data (tanpa augmentation - transfer learning sudah kuat)
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(
        BATCH_SIZE, use_augmentation=False
    )
    
    # 2. Inisialisasi Model Transfer Learning
    print(f"\n🚀 Loading {MODEL_NAME.upper()} with pretrained={'ImageNet' if USE_PRETRAINED else 'Random'} weights...")
    model = get_model(
        model_name=MODEL_NAME,
        pretrained=USE_PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    # Hitung jumlah parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 3. Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer dengan weight decay untuk regularisasi
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning Rate Scheduler - ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    # Early Stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # History tracking
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n" + "="*60)
    print(f"  🔬 TRANSFER LEARNING - {MODEL_NAME.upper()}")
    print("="*60)
    print(f"Hyperparameters:")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Pretrained: {'✅ ImageNet' if USE_PRETRAINED else '❌ Random Init'}")
    print(f"  - Freeze Backbone: {'✅ Yes' if FREEZE_BACKBONE else '❌ No (Full Fine-tuning)'}")
    print(f"  - Epochs: {EPOCHS} (max)")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Dropout: {DROPOUT_RATE}")
    print(f"  - Weight Decay: {WEIGHT_DECAY}")
    print(f"  - Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print("="*60)
    print()
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Validation Phase ---
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
        
        # Save history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Early Stopping Logic
        improvement = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improvement = "⭐ NEW BEST!"
        else:
            patience_counter += 1
            improvement = f"[Patience: {patience_counter}/{EARLY_STOP_PATIENCE}]"
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% {improvement}")
        
        # Check early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early Stopping triggered at epoch {epoch+1}")
            print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model with Val Acc: {best_val_acc:.2f}%")
    
    print("--- Training Selesai ---")
    
    # Save best model
    model_filename = f'best_model_{MODEL_NAME}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"✅ Model terbaik disimpan sebagai '{model_filename}'")
    
    # Plot training history
    plot_training_history(
        train_losses_history,
        val_losses_history,
        train_accs_history,
        val_accs_history
    )
    print("\nPlot disimpan sebagai 'training_history.png'")
    
    # Visualize predictions
    visualize_random_val_predictions(model, val_loader, num_classes=1, count=10)
    print("\nVisualisasi prediksi validation disimpan sebagai 'val_predictions.png'")

if __name__ == '__main__':
    train()
