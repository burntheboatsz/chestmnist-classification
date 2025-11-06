# train_kfold_ensemble.py
"""
K-Fold Cross-Validation Training for DenseNet121
Target: 94%+ accuracy through ensemble of 5 models
Each fold trains a separate model, then we ensemble all predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Define paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAINED_MODELS_DIR = PROJECT_DIR / 'trained_models'
RESULTS_DIR = PROJECT_DIR / 'results'

# Create directories
TRAINED_MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from data.datareader_highres import FilteredBinaryDataset, TRAIN_TRANSFORM, VAL_TRANSFORM
from models.model_densenet import get_densenet_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100
PATIENCE = 25
GRADIENT_ACCUMULATION = 2

print("=" * 70)
print("K-FOLD CROSS-VALIDATION TRAINING - TARGET: 94%+")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Number of Folds: {N_FOLDS}")
print(f"Batch Size: {BATCH_SIZE} (Effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
print(f"Max Epochs per Fold: {EPOCHS}")
print(f"Patience: {PATIENCE}")
print("=" * 70)

if torch.cuda.is_available():
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

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
            labels = labels.to(device).float().view(-1)
            
            outputs = model(images).view(-1)
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
    
    return val_loss, accuracy, precision, recall, f1, auc, all_probs

# ==================== TRAINING FUNCTION ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = images.to(device)
        labels = labels.to(device).float().view(-1)
        
        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)
        
        # Gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION
        loss.backward()
        
        if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * GRADIENT_ACCUMULATION
        
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

# ==================== MAIN TRAINING ====================
def main():
    # Load full training dataset
    print("\n📊 Loading ChestMNIST dataset...")
    full_train_dataset = FilteredBinaryDataset(split='train', transform=TRAIN_TRANSFORM)
    val_dataset = FilteredBinaryDataset(split='val', transform=VAL_TRANSFORM)
    
    print(f"✓ Total training samples: {len(full_train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Prepare validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # K-Fold setup
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    indices = np.arange(len(full_train_dataset))
    
    # Store models and their validation predictions
    fold_models = []
    fold_val_probs = []
    fold_results = []
    
    # Train each fold
    for fold, (train_idx, val_fold_idx) in enumerate(kfold.split(indices)):
        print("\n" + "=" * 70)
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print("=" * 70)
        
        # Create data loaders for this fold
        train_subset = Subset(full_train_dataset, train_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"  Training samples: {len(train_subset)}")
        
        # Create model
        model = get_densenet_model(
            model_type='standard',
            num_classes=1,
            dropout_rate=0.5,
            freeze_backbone=False
        ).to(DEVICE)
        
        # Load pretrained weights from best model
        try:
            model.load_state_dict(torch.load(TRAINED_MODELS_DIR / 'best_densenet_improved.pth'))
            print("  ✓ Loaded improved model weights as initialization")
        except:
            try:
                model.load_state_dict(torch.load(TRAINED_MODELS_DIR / 'best_densenet_model.pth'))
                print("  ✓ Loaded original model weights as initialization")
            except:
                print("  ⚠️ Training from scratch")
        
        # Optimizer and loss
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(DEVICE))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=False,
            min_lr=1e-7
        )
        
        # Training loop
        best_val_acc = 0.0
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            
            # Validate
            val_loss, val_acc, prec, rec, f1, auc, _ = validate(model, val_loader, criterion, DEVICE)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            print(f"\n  Epoch {epoch+1}/{EPOCHS}")
            print(f"    Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"    Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_auc = auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"    ✓ Best model updated (Acc: {val_acc*100:.2f}%)")
                
                if val_acc >= 0.90:
                    print(f"    🎉 90% REACHED!")
                if val_acc >= 0.94:
                    print(f"    🎊 94% REACHED!")
            else:
                patience_counter += 1
                print(f"    [Patience: {patience_counter}/{PATIENCE}]")
            
            # Early stopping
            if patience_counter >= PATIENCE:
                print(f"\n  ⚠️ Early stopping at epoch {epoch+1}")
                break
            
            # Stop if 95% reached
            if val_acc >= 0.95:
                print(f"\n  🎊 95% REACHED! Stopping fold training.")
                break
        
        # Load best model and save
        model.load_state_dict(best_model_state)
        model_path = TRAINED_MODELS_DIR / f'densenet_kfold_{fold+1}.pth'
        torch.save(best_model_state, model_path)
        
        # Get predictions on validation set
        _, val_acc, prec, rec, f1, auc, val_probs = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\n  ✅ Fold {fold+1} completed!")
        print(f"    Best Val Acc: {best_val_acc*100:.2f}%")
        print(f"    Final Metrics: Acc={val_acc*100:.2f}%, Prec={prec*100:.2f}%, Rec={rec*100:.2f}%, F1={f1*100:.2f}%, AUC={auc*100:.2f}%")
        
        # Store results
        fold_models.append(model)
        fold_val_probs.append(val_probs)
        fold_results.append({
            'fold': fold + 1,
            'accuracy': val_acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        })
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # ==================== ENSEMBLE EVALUATION ====================
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION - COMBINING ALL FOLDS")
    print("=" * 70)
    
    # Get labels
    all_labels = []
    for images, labels in val_loader:
        all_labels.extend(labels.numpy().flatten())
    all_labels = np.array(all_labels)
    
    # Average predictions
    avg_probs = np.mean(fold_val_probs, axis=0)
    
    # Find best threshold
    best_threshold = 0.5
    best_acc = 0.0
    best_metrics = None
    
    print("\n🔍 Optimizing ensemble threshold...")
    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (avg_probs >= threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            prec = precision_score(all_labels, preds, zero_division=0)
            rec = recall_score(all_labels, preds, zero_division=0)
            f1 = f1_score(all_labels, preds, zero_division=0)
            auc = roc_auc_score(all_labels, avg_probs)
            cm = confusion_matrix(all_labels, preds)
            best_metrics = (prec, rec, f1, auc, cm)
    
    prec, rec, f1, auc, cm = best_metrics
    
    print("\n" + "=" * 70)
    print("🎯 FINAL K-FOLD ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"\nIndividual Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Acc={result['accuracy']*100:.2f}%, "
              f"F1={result['f1']*100:.2f}%, AUC={result['auc']*100:.2f}%")
    
    avg_fold_acc = np.mean([r['accuracy'] for r in fold_results])
    print(f"\nAverage Fold Accuracy: {avg_fold_acc*100:.2f}%")
    
    print(f"\n{'=' * 70}")
    print(f"ENSEMBLE PERFORMANCE (Threshold: {best_threshold:.2f})")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Cardio  Pneumo")
    print(f"Actual Cardio    {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"       Pneumo    {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    if best_acc >= 0.94:
        print(f"\n🎊 EXCELLENT! 94% TARGET REACHED!")
        print(f"   Accuracy: {best_acc*100:.2f}%")
    elif best_acc >= 0.90:
        print(f"\n🎉 TARGET 90% REACHED!")
        print(f"   Gap to 94%: {(0.94 - best_acc)*100:.2f}%")
    else:
        print(f"\n⚠️ Target not reached")
        print(f"   Current: {best_acc*100:.2f}%")
        print(f"   Gap to 94%: {(0.94 - best_acc)*100:.2f}%")
    
    # Save ensemble predictions info
    ensemble_info = {
        'threshold': best_threshold,
        'accuracy': best_acc,
        'fold_results': fold_results
    }
    
    import pickle
    with open(TRAINED_MODELS_DIR / 'kfold_ensemble_info.pkl', 'wb') as f:
        pickle.dump(ensemble_info, f)
    
    print(f"\n✅ K-Fold training completed!")
    print(f"✓ {N_FOLDS} models saved in {TRAINED_MODELS_DIR}")
    print(f"✓ Ensemble info saved")
    print("=" * 70)

if __name__ == '__main__':
    main()
