# train_densenet.py
"""
Training script for DenseNet121 on ChestMNIST
Two-stage training strategy:
1. Train only classifier head (frozen backbone) - 10 epochs
2. Fine-tune entire network - 50 epochs

Target: 90%+ validation accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Define paths relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAINED_MODELS_DIR = PROJECT_DIR / 'trained_models'
RESULTS_DIR = PROJECT_DIR / 'results'

# Create directories if they don't exist
TRAINED_MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from data.datareader_highres import get_data_loaders
from models.model_densenet import get_densenet_model
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 128  # Higher resolution for better feature extraction
BATCH_SIZE = 16  # Smaller batch for larger model
LEARNING_RATE_STAGE1 = 0.001  # Higher LR for classifier training
LEARNING_RATE_STAGE2 = 0.0001  # Lower LR for fine-tuning
DROPOUT_RATE = 0.5  # Strong regularization
WEIGHT_DECAY = 1e-4
EPOCHS_STAGE1 = 10  # Train classifier only
EPOCHS_STAGE2 = 50  # Full fine-tuning
EARLY_STOP_PATIENCE = 10
MODEL_TYPE = 'standard'  # 'standard' or 'attention'
USE_AUGMENTATION = True  # Medical-safe augmentation

# Force GPU
if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU is required for DenseNet training!")

print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)
print(f"✅ GPU Active: {torch.cuda.get_device_name(0)}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   PyTorch Version: {torch.__version__}")
print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 60)

# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float()
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images).unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            outputs = model(images).unsqueeze(1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    epoch_acc = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    return epoch_loss, epoch_acc, precision, recall, f1, auc


def plot_training_history(history, save_path=None):
    """Plot and save training history"""
    if save_path is None:
        save_path = RESULTS_DIR / 'densenet_training_history.png'
    else:
        save_path = Path(save_path)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='d', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC
    axes[1, 1].plot(history['val_auc'], label='Val AUC', marker='*', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC-ROC')
    axes[1, 1].set_title('Validation AUC-ROC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_path}")
    plt.close()


# ==================== MAIN TRAINING ====================

def main():
    print("\n" + "=" * 60)
    print(f"DENSENET121 MEDICAL TRAINING - TARGET: 90%+")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model Type: {MODEL_TYPE.upper()}")
    print(f"Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Data Augmentation: {'ACTIVE' if USE_AUGMENTATION else 'INACTIVE'}")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading ChestMNIST dataset (high-resolution)...")
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(
        batch_size=BATCH_SIZE,
        use_augmentation=USE_AUGMENTATION
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n🔧 Creating DenseNet121 model...")
    model = get_densenet_model(
        model_type=MODEL_TYPE,
        num_classes=1,
        dropout_rate=DROPOUT_RATE,
        freeze_backbone=True  # Start with frozen backbone
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_auc': []
    }
    
    best_val_acc = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    
    # ==================== STAGE 1: Train Classifier Only ====================
    print("\n" + "=" * 60)
    print("STAGE 1: Training Classifier Head Only (Backbone Frozen)")
    print("=" * 60)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_STAGE1,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    for epoch in range(EPOCHS_STAGE1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, precision, recall, f1, auc = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['val_auc'].append(auc)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS_STAGE1}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = auc
            torch.save(model.state_dict(), TRAINED_MODELS_DIR / 'best_densenet_stage1.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.4f})")
    
    # Load best Stage 1 model
    model.load_state_dict(torch.load(TRAINED_MODELS_DIR / 'best_densenet_stage1.pth'))
    print(f"\n✓ Stage 1 completed! Best Val Acc: {best_val_acc:.4f}, AUC: {best_val_auc:.4f}")
    
    # ==================== STAGE 2: Full Fine-tuning ====================
    print("\n" + "=" * 60)
    print("STAGE 2: Full Fine-tuning (All Parameters Trainable)")
    print("=" * 60)
    
    # Unfreeze backbone
    if hasattr(model, 'unfreeze_backbone'):
        model.unfreeze_backbone()
    else:
        for param in model.parameters():
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ All parameters unfrozen")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # New optimizer with lower learning rate
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE_STAGE2,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    patience_counter = 0
    
    for epoch in range(EPOCHS_STAGE2):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, precision, recall, f1, auc = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['val_auc'].append(auc)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS_STAGE2}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), TRAINED_MODELS_DIR / 'best_densenet_model.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  [Patience: {patience_counter}/{EARLY_STOP_PATIENCE}]")
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
    
    # ==================== FINAL EVALUATION ====================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(TRAINED_MODELS_DIR / 'best_densenet_model.pth'))
    val_loss, val_acc, precision, recall, f1, auc = validate(model, val_loader, criterion, DEVICE)
    
    print(f"\n🎯 Best DenseNet121 Model Performance:")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    
    # Check if target reached
    if val_acc >= 0.90:
        print(f"\n🎉 TARGET REACHED! Validation Accuracy: {val_acc*100:.2f}% >= 90%")
    else:
        print(f"\n⚠️ Target not reached. Current: {val_acc*100:.2f}%, Target: 90%")
        print(f"   Gap: {(0.90 - val_acc)*100:.2f}%")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n✅ DenseNet training completed!")
    print(f"✓ Best model saved as '{TRAINED_MODELS_DIR / 'best_densenet_model.pth'}'")
    print(f"✓ Training history saved as '{RESULTS_DIR / 'densenet_training_history.png'}'")


if __name__ == '__main__':
    main()
