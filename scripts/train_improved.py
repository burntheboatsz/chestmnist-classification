# train_improved.py
# Training dengan improved dataset (balanced, 224x224, heavy augmentation)

import torch
import torch.nn as nn
import torch.optim as optim
from datareader_augmented import get_data_loaders
from model import SimpleCNN
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 30
BATCH_SIZE = 16  # Reduced karena 224x224 lebih besar (memory)
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.4  # Slightly higher karena dataset lebih besar
EARLY_STOP_PATIENCE = 7
USE_AUGMENTATION = True  # Gunakan heavy augmentation
USE_BALANCING = True  # Gunakan balanced sampling
USE_ALBUMENTATIONS = True  # Gunakan albumentations library

def train():
    # Setup Device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Memuat Improved Dataset
    print("\n" + "="*70)
    print("📦 LOADING IMPROVED DATASET")
    print("="*70)
    
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(
        batch_size=BATCH_SIZE,
        use_augmentation=USE_AUGMENTATION,
        use_balancing=USE_BALANCING,
        use_albumentations=USE_ALBUMENTATIONS
    )
    
    # 2. Inisialisasi Model untuk 224x224 input
    print("\n" + "="*70)
    print("🏗️  BUILDING MODEL")
    print("="*70)
    
    model = SimpleCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        input_size=224  # Important: 224x224 input
    ).to(device)
    
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 3. Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
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
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING WITH IMPROVED DATASET")
    print("="*70)
    print(f"Hyperparameters:")
    print(f"  - Epochs: {EPOCHS} (max)")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Dropout: {DROPOUT_RATE}")
    print(f"  - Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"  - Input Resolution: 224x224")
    print(f"  - Augmentation: {'Heavy (Albumentations)' if USE_AUGMENTATION else 'None'}")
    print(f"  - Class Balancing: {'Enabled' if USE_BALANCING else 'Disabled'}")
    print("="*70 + "\n")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
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
                labels = labels.to(device)
                
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
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    
    # Save best model
    torch.save(model.state_dict(), 'best_model_improved.pth')
    print("✅ Model terbaik disimpan sebagai 'best_model_improved.pth'")
    
    # Plot training history
    plot_training_history(
        train_losses_history,
        val_losses_history,
        train_accs_history,
        val_accs_history
    )
    print("\n✅ Plot disimpan sebagai 'training_history.png'")
    
    # Visualize predictions
    visualize_random_val_predictions(model, val_loader, num_classes=1, count=10)
    print("✅ Visualisasi prediksi validation disimpan sebagai 'val_predictions.png'")
    
    # Print final results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Training Epochs: {epoch+1}")
    print("="*70)
    
    # Compare with baseline
    baseline_acc = 83.72  # SimpleCNN 128x128
    improvement_pct = best_val_acc - baseline_acc
    print(f"\n📈 Improvement over baseline (83.72%):")
    print(f"   {improvement_pct:+.2f}% {'✅' if improvement_pct > 0 else '❌'}")
    
    if best_val_acc >= 88:
        print("\n🎉 EXCELLENT! Target 88%+ achieved!")
    elif best_val_acc >= 85:
        print("\n✅ GOOD! Close to target, consider ensemble for 88%+")
    else:
        print("\n⚠️  Below target. Consider:")
        print("   1. Train longer (increase patience)")
        print("   2. Try DenseNet121 with medical pretraining")
        print("   3. Download external data (NIH ChestX-ray14)")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    train()
