# optimize_threshold.py
"""
Optimize prediction threshold to maximize accuracy
Sometimes default 0.5 threshold is not optimal
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
print("THRESHOLD OPTIMIZATION FOR DENSENET121")
print("=" * 70)
print(f"Device: {DEVICE}")
print("=" * 70)

def evaluate_with_threshold(model, val_loader, threshold, device):
    """Evaluate model with custom threshold"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, auc, cm

# Load data
print("\n📊 Loading validation dataset...")
_, val_loader, _, _ = get_data_loaders(batch_size=16, use_augmentation=False)
print(f"✓ Validation samples: {len(val_loader.dataset)}")

# Test both models
models_to_test = [
    ('Original Model (89.51%)', TRAINED_MODELS_DIR / 'best_densenet_model.pth'),
    ('Improved Model (89.84%)', TRAINED_MODELS_DIR / 'best_densenet_improved.pth')
]

for model_name, model_path in models_to_test:
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 70}")
    
    # Load model
    model = get_densenet_model(
        model_type='standard',
        num_classes=1,
        dropout_rate=0.5,
        freeze_backbone=False
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("\n🔍 Testing different thresholds...")
    
    best_threshold = 0.5
    best_accuracy = 0.0
    best_metrics = None
    
    results = []
    
    # Test thresholds from 0.3 to 0.7
    for threshold in np.arange(0.30, 0.71, 0.01):
        acc, prec, rec, f1, auc, cm = evaluate_with_threshold(
            model, val_loader, threshold, DEVICE
        )
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'cm': cm
        })
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold
            best_metrics = {
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'cm': cm
            }
    
    # Display results
    print(f"\n📊 Results Summary:")
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    
    # Show top 10 thresholds
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:10]
    for r in sorted_results:
        mark = " ← BEST" if r['threshold'] == best_threshold else ""
        print(f"{r['threshold']:<12.2f} {r['accuracy']*100:<11.2f}% {r['precision']*100:<11.2f}% "
              f"{r['recall']*100:<11.2f}% {r['f1']*100:<11.2f}%{mark}")
    
    print(f"\n{'=' * 70}")
    print(f"🎯 BEST THRESHOLD: {best_threshold:.2f}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"  Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {best_metrics['f1']:.4f} ({best_metrics['f1']*100:.2f}%)")
    print(f"  AUC-ROC:   {best_metrics['auc']:.4f} ({best_metrics['auc']*100:.2f}%)")
    
    print(f"\n📈 Confusion Matrix (Threshold: {best_threshold:.2f}):")
    cm = best_metrics['cm']
    print(f"                 Predicted")
    print(f"               Cardio  Pneumo")
    print(f"Actual Cardio    {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"       Pneumo    {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    if best_accuracy >= 0.90:
        print(f"\n🎉 TARGET 90% REACHED with threshold {best_threshold:.2f}!")
    else:
        gap = (0.90 - best_accuracy) * 100
        print(f"\n⚠️ Gap to 90%: {gap:.2f}%")

print("\n" + "=" * 70)
print("✅ Threshold optimization completed!")
print("=" * 70)
