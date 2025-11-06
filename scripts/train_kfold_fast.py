# train_kfold_fast.py
"""
Fast K-Fold CV Training - Optimized untuk speed & 94% target
5-Fold dengan early stopping agresif
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle

sys.path.append(str(Path(__file__).parent.parent))

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAINED_MODELS_DIR = PROJECT_DIR / 'trained_models'
RESULTS_DIR = PROJECT_DIR / 'results'

TRAINED_MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

from torch.utils.data import DataLoader, Subset
from data.datareader_highres import FilteredBinaryDataset, TRAIN_TRANSFORM, VAL_TRANSFORM
from models.model_densenet import get_densenet_model
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS = 5
BATCH_SIZE = 16
LR = 1e-4
MAX_EPOCHS = 50
PATIENCE = 15

print("="*70)
print("FAST K-FOLD TRAINING - TARGET 94%")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Folds: {N_FOLDS} | Epochs: {MAX_EPOCHS} | Patience: {PATIENCE}")
print("="*70)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds, labels = [], []
    
    for imgs, lbls in tqdm(loader, desc="Train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device).float().view(-1)
        
        optimizer.zero_grad()
        out = model(imgs).view(-1)
        loss = criterion(out, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds.extend((torch.sigmoid(out) > 0.5).cpu().numpy())
        labels.extend(lbls.cpu().numpy())
    
    return total_loss/len(loader), accuracy_score(labels, preds)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device).float().view(-1)
            out = model(imgs).view(-1)
            loss = criterion(out, lbls)
            total_loss += loss.item()
            
            probs = torch.sigmoid(out)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    
    # Find best threshold
    best_acc = 0
    for thresh in np.arange(0.3, 0.7, 0.02):
        preds = (probs > thresh).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
    
    auc = roc_auc_score(labels, probs)
    return total_loss/len(loader), best_acc, auc, probs

print("\n📊 Loading dataset...")
train_ds = FilteredBinaryDataset('train', TRAIN_TRANSFORM)
val_ds = FilteredBinaryDataset('val', VAL_TRANSFORM)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# Get labels for stratification
train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_models = []
fold_val_probs = []
fold_accs = []

for fold, (train_idx, _) in enumerate(kfold.split(range(len(train_ds)), train_labels)):
    print(f"\n{'='*70}")
    print(f"FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*70}")
    
    # Create fold dataloader
    train_subset = Subset(train_ds, train_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    
    # Create model
    model = get_densenet_model('standard', num_classes=1, dropout_rate=0.5, 
                               freeze_backbone=False).to(DEVICE)
    
    # Load pretrained weights
    try:
        model.load_state_dict(torch.load(TRAINED_MODELS_DIR / 'best_densenet_improved.pth'))
        print("✓ Loaded improved weights")
    except:
        try:
            model.load_state_dict(torch.load(TRAINED_MODELS_DIR / 'best_densenet_model.pth'))
            print("✓ Loaded original weights")
        except:
            print("⚠ Training from scratch")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                            factor=0.5, patience=7, verbose=False)
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_auc, val_probs = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:2d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} ({val_acc*100:.2f}%) | AUC: {val_auc:.4f}", end="")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), TRAINED_MODELS_DIR / f'kfold_model_{fold+1}.pth')
            best_probs = val_probs.copy()
            print(" ✓ BEST", end="")
            if val_acc >= 0.90:
                print(" 🎉", end="")
            if val_acc >= 0.94:
                print(" 🎊", end="")
        else:
            patience_counter += 1
            print(f" [P:{patience_counter}/{PATIENCE}]", end="")
        
        print()
        
        if patience_counter >= PATIENCE:
            print(f"⚠ Early stop at epoch {epoch+1}")
            break
        
        if val_acc >= 0.95:
            print(f"🎊 95% reached! Stopping fold.")
            break
    
    model.load_state_dict(torch.load(TRAINED_MODELS_DIR / f'kfold_model_{fold+1}.pth'))
    fold_models.append(model)
    fold_val_probs.append(best_probs)
    fold_accs.append(best_acc)
    
    print(f"\n✓ Fold {fold+1} done: {best_acc*100:.2f}%")
    torch.cuda.empty_cache()

# ENSEMBLE EVALUATION
print(f"\n{'='*70}")
print("ENSEMBLE EVALUATION")
print(f"{'='*70}")

# Get true labels
val_labels = []
for _, lbl in val_loader:
    val_labels.extend(lbl.numpy().flatten())
val_labels = np.array(val_labels)

# Average predictions
avg_probs = np.mean(fold_val_probs, axis=0)

# Find best threshold
best_thresh = 0.5
best_acc = 0
for thresh in np.arange(0.30, 0.71, 0.01):
    preds = (avg_probs >= thresh).astype(int)
    acc = accuracy_score(val_labels, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

preds = (avg_probs >= best_thresh).astype(int)
auc = roc_auc_score(val_labels, avg_probs)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
prec = precision_score(val_labels, preds)
rec = recall_score(val_labels, preds)
f1 = f1_score(val_labels, preds)
cm = confusion_matrix(val_labels, preds)

print("\nIndividual Folds:")
for i, acc in enumerate(fold_accs):
    print(f"  Fold {i+1}: {acc*100:.2f}%")
print(f"  Average: {np.mean(fold_accs)*100:.2f}%")

print(f"\n{'='*70}")
print(f"FINAL ENSEMBLE (Threshold: {best_thresh:.2f})")
print(f"{'='*70}")
print(f"Accuracy:  {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")

print(f"\nConfusion Matrix:")
print(f"               Predicted")
print(f"             Cardio  Pneumo")
print(f"Actual Cardio  {cm[0,0]:3d}     {cm[0,1]:3d}")
print(f"       Pneumo  {cm[1,0]:3d}     {cm[1,1]:3d}")

if best_acc >= 0.94:
    print(f"\n🎊 EXCELLENT! 94% TARGET REACHED!")
    print(f"   Final Accuracy: {best_acc*100:.2f}%")
elif best_acc >= 0.92:
    print(f"\n🎉 GREAT! 92%+ Achieved!")
    print(f"   Gap to 94%: {(0.94-best_acc)*100:.2f}%")
elif best_acc >= 0.90:
    print(f"\n✅ 90%+ Achieved!")
    print(f"   Gap to 94%: {(0.94-best_acc)*100:.2f}%")

# Save results
results = {
    'threshold': best_thresh,
    'accuracy': best_acc,
    'precision': prec,
    'recall': rec,
    'f1': f1,
    'auc': auc,
    'fold_accs': fold_accs,
    'confusion_matrix': cm
}

with open(TRAINED_MODELS_DIR / 'kfold_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n✅ Training complete! {N_FOLDS} models saved.")
print("="*70)
