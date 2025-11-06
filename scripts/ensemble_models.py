# ensemble_models.py
"""
Ensemble prediction from multiple DenseNet models
Combines predictions from original and improved models
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Define paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAINED_MODELS_DIR = PROJECT_DIR / 'trained_models'

from data.datareader_highres import get_data_loaders
from models.model_densenet import get_densenet_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("ENSEMBLE PREDICTION - COMBINING MULTIPLE DENSENET MODELS")
print("=" * 70)
print(f"Device: {DEVICE}")
print("=" * 70)

# Load data
print("\n📊 Loading validation dataset...")
_, val_loader, _, _ = get_data_loaders(batch_size=16, use_augmentation=False)
print(f"✓ Validation samples: {len(val_loader.dataset)}")

# Load models
print("\n🔧 Loading models...")
models = []
model_names = []

model_paths = [
    ('Original Model (89.51%)', TRAINED_MODELS_DIR / 'best_densenet_model.pth'),
    ('Improved Model (89.84%)', TRAINED_MODELS_DIR / 'best_densenet_improved.pth'),
    ('Stage 1 Model', TRAINED_MODELS_DIR / 'best_densenet_stage1.pth'),
]

for name, path in model_paths:
    if path.exists():
        print(f"  Loading: {name}")
        model = get_densenet_model(
            model_type='standard',
            num_classes=1,
            dropout_rate=0.5,
            freeze_backbone=False
        ).to(DEVICE)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)
        model_names.append(name)
    else:
        print(f"  ⚠️ Skipping {name} (file not found)")

print(f"\n✓ Loaded {len(models)} models for ensemble")

# Ensemble prediction with different strategies
print("\n" + "=" * 70)
print("ENSEMBLE STRATEGIES")
print("=" * 70)

def get_predictions(models, val_loader, device):
    """Get predictions from all models"""
    all_model_probs = [[] for _ in range(len(models))]
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            for i, model in enumerate(models):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                all_model_probs[i].extend(probs.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to arrays
    all_model_probs = [np.array(probs).flatten() for probs in all_model_probs]
    all_labels = np.array(all_labels).flatten()
    
    return all_model_probs, all_labels

# Get predictions from all models
print("\n🔍 Getting predictions from all models...")
all_probs, all_labels = get_predictions(models, val_loader, DEVICE)

# Strategy 1: Simple Average
print("\n" + "=" * 70)
print("Strategy 1: SIMPLE AVERAGE")
print("=" * 70)

avg_probs = np.mean(all_probs, axis=0)

best_threshold = 0.5
best_accuracy = 0.0
best_metrics = None

for threshold in np.arange(0.30, 0.71, 0.01):
    preds = (avg_probs >= threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        auc = roc_auc_score(all_labels, avg_probs)
        cm = confusion_matrix(all_labels, preds)
        best_metrics = (prec, rec, f1, auc, cm)

prec, rec, f1, auc, cm = best_metrics
print(f"\nBest Threshold: {best_threshold:.2f}")
print(f"  Accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")

print(f"\n📈 Confusion Matrix:")
print(f"                 Predicted")
print(f"               Cardio  Pneumo")
print(f"Actual Cardio    {cm[0,0]:3d}     {cm[0,1]:3d}")
print(f"       Pneumo    {cm[1,0]:3d}     {cm[1,1]:3d}")

if best_accuracy >= 0.90:
    print(f"\n🎉 TARGET 90% REACHED!")
    target_90_reached = True
else:
    gap = (0.90 - best_accuracy) * 100
    print(f"\n⚠️ Gap to 90%: {gap:.2f}%")
    target_90_reached = False

# Strategy 2: Weighted Average (give more weight to better model)
print("\n" + "=" * 70)
print("Strategy 2: WEIGHTED AVERAGE (More weight to improved model)")
print("=" * 70)

# Weights based on individual accuracy
if len(models) == 3:
    weights = np.array([0.3, 0.5, 0.2])  # More weight to improved model
elif len(models) == 2:
    weights = np.array([0.4, 0.6])  # More weight to improved model
else:
    weights = np.ones(len(models)) / len(models)

weighted_probs = np.average(all_probs, axis=0, weights=weights)

best_threshold = 0.5
best_accuracy = 0.0
best_metrics = None

for threshold in np.arange(0.30, 0.71, 0.01):
    preds = (weighted_probs >= threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        auc = roc_auc_score(all_labels, weighted_probs)
        cm = confusion_matrix(all_labels, preds)
        best_metrics = (prec, rec, f1, auc, cm)

prec, rec, f1, auc, cm = best_metrics
print(f"\nBest Threshold: {best_threshold:.2f}")
print(f"  Accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")

print(f"\n📈 Confusion Matrix:")
print(f"                 Predicted")
print(f"               Cardio  Pneumo")
print(f"Actual Cardio    {cm[0,0]:3d}     {cm[0,1]:3d}")
print(f"       Pneumo    {cm[1,0]:3d}     {cm[1,1]:3d}")

if best_accuracy >= 0.90:
    print(f"\n🎉 TARGET 90% REACHED!")
    target_90_reached = True
else:
    gap = (0.90 - best_accuracy) * 100
    print(f"\n⚠️ Gap to 90%: {gap:.2f}%")

# Strategy 3: Voting
print("\n" + "=" * 70)
print("Strategy 3: MAJORITY VOTING")
print("=" * 70)

best_accuracy = 0.0
best_metrics = None

for threshold in np.arange(0.30, 0.71, 0.01):
    # Get predictions from each model
    model_preds = [(probs >= threshold).astype(int) for probs in all_probs]
    # Majority vote
    voted_preds = np.round(np.mean(model_preds, axis=0)).astype(int)
    
    acc = accuracy_score(all_labels, voted_preds)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold
        prec = precision_score(all_labels, voted_preds, zero_division=0)
        rec = recall_score(all_labels, voted_preds, zero_division=0)
        f1 = f1_score(all_labels, voted_preds, zero_division=0)
        auc = roc_auc_score(all_labels, avg_probs)
        cm = confusion_matrix(all_labels, voted_preds)
        best_metrics = (prec, rec, f1, auc, cm)

prec, rec, f1, auc, cm = best_metrics
print(f"\nBest Threshold: {best_threshold:.2f}")
print(f"  Accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")

print(f"\n📈 Confusion Matrix:")
print(f"                 Predicted")
print(f"               Cardio  Pneumo")
print(f"Actual Cardio    {cm[0,0]:3d}     {cm[0,1]:3d}")
print(f"       Pneumo    {cm[1,0]:3d}     {cm[1,1]:3d}")

if best_accuracy >= 0.90:
    print(f"\n🎉 TARGET 90% REACHED!")
else:
    gap = (0.90 - best_accuracy) * 100
    print(f"\n⚠️ Gap to 90%: {gap:.2f}%")

print("\n" + "=" * 70)
print("✅ Ensemble evaluation completed!")
print("=" * 70)
