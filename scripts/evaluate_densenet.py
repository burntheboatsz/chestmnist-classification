# evaluate_densenet.py
"""
Comprehensive evaluation script for DenseNet121 model
Generates:
- Confusion Matrix
- Classification Report
- ROC Curve
- Precision-Recall Curve
- Detailed metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from datareader_highres import get_data_loaders
from model_densenet import get_densenet_model
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '../best_densenet_model.pth'
BATCH_SIZE = 16
CLASS_NAMES = ['Cardiomegaly', 'Pneumothorax']

print("=" * 70)
print("DENSENET121 MODEL EVALUATION")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print("=" * 70)

# Load data
print("\n📊 Loading validation dataset...")
_, val_loader, num_classes, in_channels = get_data_loaders(
    batch_size=BATCH_SIZE,
    use_augmentation=False  # No augmentation for evaluation
)
print(f"✓ Validation samples: {len(val_loader.dataset)}")

# Load model
print("\n🔧 Loading DenseNet121 model...")
model = get_densenet_model(
    model_type='standard',
    num_classes=1,
    dropout_rate=0.5,
    freeze_backbone=False
).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✓ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found: {MODEL_PATH}")
    print("Trying alternative path...")
    MODEL_PATH = 'best_densenet_model.pth'
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✓ Model loaded from {MODEL_PATH}")

model.eval()

# Evaluate
print("\n🔍 Evaluating model on validation set...")
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.cpu().numpy()
        
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

print("✓ Evaluation completed!")

# ==================== METRICS CALCULATION ====================
print("\n" + "=" * 70)
print("PERFORMANCE METRICS")
print("=" * 70)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

print(f"\n📊 Overall Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"  AUC-ROC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")

# ==================== CONFUSION MATRIX ====================
cm = confusion_matrix(all_labels, all_preds)
print(f"\n📈 Confusion Matrix:")
print(f"                 Predicted")
print(f"               Cardio  Pneumo")
print(f"Actual Cardio    {cm[0,0]:3d}     {cm[0,1]:3d}")
print(f"       Pneumo    {cm[1,0]:3d}     {cm[1,1]:3d}")

# Calculate per-class metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

print(f"\n📋 Detailed Metrics:")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")
print(f"\n  Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"  Specificity:          {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  PPV (Precision):      {ppv:.4f} ({ppv*100:.2f}%)")
print(f"  NPV:                  {npv:.4f} ({npv*100:.2f}%)")

# ==================== CLASSIFICATION REPORT ====================
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(all_labels, all_preds, 
                          target_names=CLASS_NAMES, 
                          digits=4))

# ==================== VISUALIZATION ====================
print("\n📊 Generating evaluation plots...")

fig = plt.figure(figsize=(20, 12))

# 1. Confusion Matrix Heatmap
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, 
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

# Add percentages
for i in range(2):
    for j in range(2):
        total = cm[i].sum()
        percentage = cm[i, j] / total * 100 if total > 0 else 0
        ax1.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

# 2. Normalized Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Percentage'})
ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
ax3.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax3.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Precision-Recall Curve
ax4 = plt.subplot(2, 3, 4)
precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(recall_curve, precision_curve)
ax4.plot(recall_curve, precision_curve, color='green', lw=2,
         label=f'PR curve (AUC = {pr_auc:.4f})')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax4.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
ax4.legend(loc="lower left", fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Metrics Bar Chart
ax5 = plt.subplot(2, 3, 5)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
bars = ax5.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
ax5.set_ylim([0, 1.0])
ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
ax5.set_title('Performance Metrics Overview', fontsize=14, fontweight='bold', pad=20)
ax5.grid(True, axis='y', alpha=0.3)
ax5.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='90% Target')
ax5.legend(fontsize=10)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{value:.4f}\n({value*100:.2f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6. Error Analysis
ax6 = plt.subplot(2, 3, 6)
error_types = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
error_counts = [tn, fp, fn, tp]
error_colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db']
bars = ax6.bar(error_types, error_counts, color=error_colors, alpha=0.8, edgecolor='black')
ax6.set_ylabel('Count', fontsize=12, fontweight='bold')
ax6.set_title('Prediction Distribution', fontsize=14, fontweight='bold', pad=20)
ax6.grid(True, axis='y', alpha=0.3)

# Add value labels
for bar, count in zip(bars, error_counts):
    height = bar.get_height()
    total = sum(error_counts)
    percentage = count / total * 100
    ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{count}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('DenseNet121 Model Evaluation - ChestMNIST Binary Classification', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('densenet_evaluation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: densenet_evaluation_matrix.png")

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(f"\n🎯 Model Performance:")
print(f"  ✓ Validation Accuracy: {accuracy*100:.2f}%")
print(f"  ✓ AUC-ROC: {roc_auc*100:.2f}%")
print(f"  ✓ F1-Score: {f1*100:.2f}%")

print(f"\n📊 Confusion Matrix Summary:")
print(f"  ✓ Correctly Classified: {tn + tp}/{len(all_labels)} ({(tn+tp)/len(all_labels)*100:.2f}%)")
print(f"  ✗ Misclassified: {fp + fn}/{len(all_labels)} ({(fp+fn)/len(all_labels)*100:.2f}%)")

print(f"\n🔍 Medical Relevance (Pneumothorax Detection):")
print(f"  ✓ Sensitivity (True Positive Rate): {sensitivity*100:.2f}%")
print(f"    → {tp}/{tp+fn} pneumothorax cases detected")
print(f"  ✓ Specificity (True Negative Rate): {specificity*100:.2f}%")
print(f"    → {tn}/{tn+fp} cardiomegaly cases correctly identified")
print(f"  ⚠️ False Negatives: {fn} ({fn/(tp+fn)*100:.2f}% missed pneumothorax)")
print(f"  ⚠️ False Positives: {fp} ({fp/(tn+fp)*100:.2f}% misdiagnosed)")

if accuracy >= 0.90:
    print(f"\n🎉 TARGET ACHIEVED! Accuracy {accuracy*100:.2f}% >= 90%")
else:
    print(f"\n⚠️ Target not reached. Current: {accuracy*100:.2f}%, Target: 90%")

print("\n" + "=" * 70)
print("✅ Evaluation completed!")
print("=" * 70)
