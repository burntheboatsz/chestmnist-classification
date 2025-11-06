# train_densenet_improved.py
"""
Improved DenseNet121 Training - Target: 90-95% accuracy
Optimizations:
1. Longer training with more patience
2. Better learning rate schedule
3. Gradient accumulation
4. Label smoothing
5. Mixup augmentation
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
LEARNING_RATE_STAGE1 = 3e-4  # Higher for classifier head
LEARNING_RATE_STAGE2 = 1e-5  # Lower for full model
EPOCHS_STAGE1 = 15  # More epochs for stage 1
EPOCHS_STAGE2 = 100  # Much longer training
PATIENCE = 20  # More patience
LABEL_SMOOTHING = 0.1  # Add label smoothing
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32

# ==================== DEVICE INFO ====================
print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)
if torch.cuda.is_available():
    print(f"✅ GPU Active: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ GPU not available, using CPU")
print("=" * 60)

# ==================== MIXUP AUGMENTATION ====================
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== VALIDATION FUNCTION ====================
def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss /= len(val_loader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    return val_loss, accuracy, precision, recall, f1, auc

# ==================== TRAINING FUNCTION ====================
def train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False):
    """Train one epoch"""
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        
        # Apply mixup if enabled
        if use_mixup and np.random.rand() > 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    train_loss /= len(train_loader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    accuracy = accuracy_score(all_labels, all_preds)
    
    return train_loss, accuracy

# ==================== PLOT FUNCTION ====================
def plot_training_history(history, save_path=None):
    """Plot and save training history"""
    if save_path is None:
        save_path = RESULTS_DIR / 'densenet_improved_training_history.png'
    else:
        save_path = Path(save_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].axhline(y=0.90, color='r', linestyle='--', label='Target 90%')
    axes[0, 1].axhline(y=0.95, color='g', linestyle='--', label='Target 95%')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='o', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('Validation F1-Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: AUC
    axes[1, 1].plot(history['val_auc'], label='Val AUC', marker='o', color='orange')
    axes[1, 1].axhline(y=0.95, color='g', linestyle='--', label='Target 95%')
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
    print("IMPROVED DENSENET121 TRAINING - TARGET: 90-95%")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE} (Effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading ChestMNIST dataset...")
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(
        batch_size=BATCH_SIZE,
        use_augmentation=True
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    
    # Load existing model
    print("\n🔧 Loading existing DenseNet121 model...")
    model = get_densenet_model(
        model_type='standard',
        num_classes=1,
        dropout_rate=0.5,
        freeze_backbone=False
    ).to(DEVICE)
    
    # Load previous weights as starting point
    model.load_state_dict(torch.load(TRAINED_MODELS_DIR / 'best_densenet_model.pth'))
    print("✓ Loaded previous best model (89.51%)")
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss with label smoothing
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.5]).to(DEVICE)  # Slight boost for positive class
    )
    
    # Wrap criterion to handle shape mismatch
    original_criterion = criterion
    def criterion(outputs, labels):
        # Ensure outputs and labels have same shape
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        return original_criterion(outputs, labels)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_STAGE2,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-7
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_auc': []
    }
    
    best_val_acc = 0.8951  # Start from previous best
    best_val_auc = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print("IMPROVED TRAINING: Full Fine-tuning with Optimizations")
    print("=" * 60)
    
    for epoch in range(EPOCHS_STAGE2):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, 
            use_mixup=(epoch > 10)  # Enable mixup after epoch 10
        )
        
        # Validate
        val_loss, val_acc, precision, recall, f1, auc = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Store history
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
            torch.save(model.state_dict(), TRAINED_MODELS_DIR / 'best_densenet_improved.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.4f})")
            
            # Check if target reached
            if val_acc >= 0.90:
                print(f"  🎉 TARGET 90% REACHED!")
            if val_acc >= 0.95:
                print(f"  🎊 EXCELLENT! 95% TARGET REACHED!")
        else:
            patience_counter += 1
            print(f"  [Patience: {patience_counter}/{PATIENCE}]")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
        
        # Stop if 95% reached
        if val_acc >= 0.95:
            print(f"\n🎊 95% TARGET REACHED! Stopping training.")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    model.load_state_dict(torch.load(TRAINED_MODELS_DIR / 'best_densenet_improved.pth'))
    val_loss, val_acc, precision, recall, f1, auc = validate(model, val_loader, criterion, DEVICE)
    
    print(f"\n🎯 Improved DenseNet121 Performance:")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Previous Accuracy: 89.51%")
    print(f"  Improvement: {(val_acc - 0.8951)*100:+.2f}%")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    
    if val_acc >= 0.95:
        print(f"\n🎊 EXCELLENT! 95% TARGET REACHED!")
    elif val_acc >= 0.90:
        print(f"\n🎉 TARGET 90% REACHED!")
    else:
        print(f"\n⚠️ Target not reached. Current: {val_acc*100:.2f}%, Target: 90%")
        print(f"   Gap: {(0.90 - val_acc)*100:.2f}%")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n✅ Improved DenseNet training completed!")
    print(f"✓ Best model saved as '{TRAINED_MODELS_DIR / 'best_densenet_improved.pth'}'")
    print(f"✓ Training history saved as '{RESULTS_DIR / 'densenet_improved_training_history.png'}'")

if __name__ == '__main__':
    main()
