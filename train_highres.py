"""
Training Script untuk High Resolution Model (128x128)

Target: 87-90% validation accuracy (naik dari 81-84%)

Improvements:
1. ✅ Resolution 28x28 -> 128x128 (+2-3% expected)
2. ✅ Medical-safe augmentation (+1-2% expected)
3. ✅ Deeper architecture (4 conv layers)
4. ✅ Channel attention mechanism
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Import custom modules
from datareader_highres import get_data_loaders
from model_highres import SimpleCNN_HighRes, AttentionCNN_HighRes

# --- GPU Configuration (FORCED) ---
if not torch.cuda.is_available():
    raise RuntimeError(
        "❌ CUDA/GPU tidak tersedia! Training HARUS menggunakan GPU.\n"
        "Pastikan environment pytorch-gpu aktif: conda activate pytorch-gpu"
    )

device = torch.device('cuda')
print(f"✅ GPU Active: {torch.cuda.get_device_name(0)}")

# --- Hyperparameters ---
BATCH_SIZE = 16  # Reduced karena resolusi lebih tinggi (128x128)
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping patience
DROPOUT_RATE = 0.4
USE_ATTENTION = False  # Set True untuk gunakan AttentionCNN

# Data augmentation
USE_AUGMENTATION = True  # Medical-safe augmentation

# Model selection
MODEL_NAME = "AttentionCNN" if USE_ATTENTION else "SimpleCNN_HighRes"

def set_seed(seed=42):
    """Set random seeds untuk reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train untuk satu epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(-1)  # Only squeeze last dimension
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validasi model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze(-1)  # Only squeeze last dimension
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return epoch_loss, accuracy, precision, recall, f1, auc, all_preds, all_labels

def plot_training_history(history, save_path='training_history_highres.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision & Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', marker='o')
    axes[1, 0].plot(history['val_recall'], label='Recall', marker='s')
    axes[1, 0].plot(history['val_f1'], label='F1 Score', marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC
    axes[1, 1].plot(history['val_auc'], label='AUC', marker='o', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Training history saved to {save_path}")

def train():
    """Main training function"""
    print("=" * 70)
    print("🚀 HIGH RESOLUTION TRAINING - TARGET 87-90% ACCURACY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Resolution: 128x128 (vs 28x28 baseline)")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Dropout: {DROPOUT_RATE}")
    print(f"Augmentation: {'ON' if USE_AUGMENTATION else 'OFF'}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 70)
    
    # Set seed
    set_seed(42)
    
    # Load data
    print("\n📊 Loading data...")
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(
        batch_size=BATCH_SIZE,
        use_augmentation=USE_AUGMENTATION
    )
    
    # Create model
    print(f"\n🏗️ Creating {MODEL_NAME}...")
    if USE_ATTENTION:
        model = AttentionCNN_HighRes(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE
        )
    else:
        model = SimpleCNN_HighRes(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE,
            input_size=128
        )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_auc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n🏋️ Starting training for {NUM_EPOCHS} epochs...")
    print("=" * 70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📍 Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Precision:  {val_precision:.4f} | Recall: {val_recall:.4f}")
        print(f"  F1 Score:   {val_f1:.4f} | AUC: {val_auc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_highres.pth')
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Target was: 87-90%")
    
    if best_val_acc >= 0.87:
        print("✅ TARGET ACHIEVED! 🎯")
    elif best_val_acc >= 0.85:
        print("⚠️ Close to target. Try attention model or more epochs.")
    else:
        print("❌ Target not met. Consider ensemble or external data.")
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    print("\n📊 Final Evaluation on Best Model...")
    model.load_state_dict(torch.load('best_model_highres.pth'))
    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, preds, labels = validate(
        model, val_loader, criterion, device
    )
    
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall:    {val_recall:.4f}")
    print(f"  F1 Score:  {val_f1:.4f}")
    print(f"  AUC:       {val_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")
    
    print("\n" + "=" * 70)
    print("💾 Saved files:")
    print("  - best_model_highres.pth (best model weights)")
    print("  - training_history_highres.png (training plots)")
    print("=" * 70)

if __name__ == "__main__":
    train()
