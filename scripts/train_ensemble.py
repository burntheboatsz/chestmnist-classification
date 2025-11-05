"""
Training Script untuk Ensemble Model
Strategi: Train SimpleCNN + ResNet18 secara terpisah, lalu combine dengan ensemble
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
from tqdm import tqdm

from datareader import get_data_loaders
from ensemble_model import EnsembleModel, AveragingEnsemble
from model import SimpleCNN
from model_resnet import get_model

# Set seed function
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Smaller LR for ensemble fine-tuning
EPOCHS = 50
INPUT_SIZE = 28
SEED = 42

# Force GPU - Raise error if CUDA not available
if not torch.cuda.is_available():
    raise RuntimeError(
        "❌ CUDA/GPU tidak tersedia!\n"
        "Ensemble training MEMERLUKAN GPU untuk performa optimal.\n"
        "Solusi:\n"
        "1. Gunakan environment: conda activate pytorch-gpu\n"
        "2. Atau jalankan: C:\\Users\\hnafi\\miniconda3\\envs\\pytorch-gpu\\python.exe train_ensemble.py"
    )

DEVICE = torch.device('cuda')

# Display device info
print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)
print(f"✅ GPU Active: {torch.cuda.get_device_name(0)}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   PyTorch Version: {torch.__version__}")
print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 60)

# Paths untuk pre-trained models
SIMPLE_MODEL_PATH = 'best_model.pth'  # SimpleCNN terbaik (81-84%)
RESNET_MODEL_PATH = 'best_model_resnet.pth'  # ResNet18 (80%)

# Set seed
set_seed(SEED)

print("=" * 60)
print("ENSEMBLE MODEL TRAINING - SimpleCNN + ResNet18")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print("=" * 60)


def evaluate_model(model, dataloader, criterion):
    """
    Evaluate model dengan metrics lengkap
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).squeeze().unsqueeze(1).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    return epoch_loss, accuracy, precision, recall, f1, auc


def train_ensemble_weights_only(model, train_loader, val_loader, epochs=10):
    """
    Stage 1: Train hanya ensemble weights (freeze base models)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Training Ensemble Weights Only")
    print("=" * 60)
    
    model.freeze_base_models()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01  # Higher LR for weights only
    )
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).squeeze().unsqueeze(1).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(
            model, val_loader, criterion
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Weights - Simple: {torch.softmax(torch.stack([model.weight_simple, model.weight_resnet]), dim=0)[0]:.3f}, "
              f"ResNet: {torch.softmax(torch.stack([model.weight_simple, model.weight_resnet]), dim=0)[1]:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_ensemble_weights.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.4f})")
    
    return history


def train_ensemble_full_finetune(model, train_loader, val_loader, epochs=30):
    """
    Stage 2: Fine-tune seluruh ensemble (unfreeze all)
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Full Ensemble Fine-tuning")
    print("=" * 60)
    
    model.unfreeze_all()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).squeeze().unsqueeze(1).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(
            model, val_loader, criterion
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_ensemble_model.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return history


def plot_training_history(history, filename='ensemble_training_history.png'):
    """
    Plot training curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training history saved to {filename}")


def main():
    # Load data
    print("\n📊 Loading ChestMNIST dataset...")
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(
        batch_size=BATCH_SIZE,
        use_augmentation=False  # No augmentation untuk ensemble
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    print(f"✓ Num classes: {num_classes}")
    print(f"✓ Input channels: {in_channels}")
    
    # Create ensemble model
    print("\n🔧 Creating Ensemble Model...")
    model = EnsembleModel(input_size=INPUT_SIZE, device=DEVICE)
    
    # Load pre-trained models
    print("\n📥 Loading pre-trained models...")
    model.load_pretrained_models(SIMPLE_MODEL_PATH, RESNET_MODEL_PATH)
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Stage 1: Train ensemble weights only
    history_stage1 = train_ensemble_weights_only(model, train_loader, val_loader, epochs=10)
    
    # Load best weights from stage 1
    model.load_state_dict(torch.load('best_ensemble_weights.pth'))
    
    # Stage 2: Full fine-tuning
    history_stage2 = train_ensemble_full_finetune(model, train_loader, val_loader, epochs=EPOCHS)
    
    # Combine histories
    full_history = {
        'train_loss': history_stage1['train_loss'] + history_stage2['train_loss'],
        'val_loss': history_stage1['val_loss'] + history_stage2['val_loss'],
        'train_acc': history_stage1['train_acc'] + history_stage2['train_acc'],
        'val_acc': history_stage1['val_acc'] + history_stage2['val_acc']
    }
    
    # Plot results
    plot_training_history(full_history)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    model.load_state_dict(torch.load('best_ensemble_model.pth'))
    criterion = nn.BCEWithLogitsLoss()
    
    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(
        model, val_loader, criterion
    )
    
    print(f"\n🎯 Best Ensemble Model Performance:")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall: {val_rec:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")
    print(f"  AUC-ROC: {val_auc:.4f}")
    
    print("\n✅ Ensemble training completed!")
    print(f"✓ Best model saved as 'best_ensemble_model.pth'")
    print(f"✓ Training history saved as 'ensemble_training_history.png'")


if __name__ == '__main__':
    main()
