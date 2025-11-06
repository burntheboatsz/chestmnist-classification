# visualize_kfold_results.py
"""
Visualize K-Fold Ensemble Results - 92.46% Achievement
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / 'results'
TRAINED_MODELS_DIR = PROJECT_DIR / 'trained_models'

# Load results
with open(TRAINED_MODELS_DIR / 'kfold_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('K-Fold Cross-Validation Results - 92.46% Accuracy Achieved!', 
             fontsize=20, fontweight='bold', y=0.98)

# 1. Individual Fold Performance
ax1 = fig.add_subplot(gs[0, 0])
folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Ensemble']
fold_accs = results['fold_accs'] + [results['accuracy']]
colors = ['#3498db']*5 + ['#e74c3c']
bars = ax1.bar(folds, [acc*100 for acc in fold_accs], color=colors, edgecolor='black', linewidth=1.5)
ax1.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='Target 90%', alpha=0.7)
ax1.axhline(y=94, color='green', linestyle='--', linewidth=2, label='Target 94%', alpha=0.7)
ax1.axhline(y=np.mean(results['fold_accs'])*100, color='purple', linestyle=':', linewidth=2, 
            label=f'Avg Fold: {np.mean(results["fold_accs"])*100:.2f}%', alpha=0.7)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Individual Fold Performance', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([80, 95])
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Metrics Comparison (Bar Chart)
ax2 = fig.add_subplot(gs[0, 1])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
values = [results['accuracy'], results['precision'], results['recall'], 
          results['f1'], results['auc']]
bars = ax2.barh(metrics, [v*100 for v in values], color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c'],
                edgecolor='black', linewidth=1.5)
ax2.axvline(x=90, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=94, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Ensemble Performance Metrics', fontsize=14, fontweight='bold')
ax2.set_xlim([85, 100])
ax2.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
             f'{width:.2f}%', ha='left', va='center', fontweight='bold', fontsize=11)

# 3. Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
cm = results['confusion_matrix']
im = ax3.imshow(cm, cmap='Blues', aspect='auto')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Cardiomegaly', 'Pneumothorax'], fontsize=11)
ax3.set_yticklabels(['Cardiomegaly', 'Pneumothorax'], fontsize=11)
ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax3.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, f'{cm[i, j]}\n({cm[i,j]/cm.sum(axis=1)[i]*100:.1f}%)',
                       ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                       fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax3)

# 4. Progress Through Models
ax4 = fig.add_subplot(gs[1, :])
model_progression = [
    ('SimpleCNN', 82.5),
    ('ResNet18', 80),
    ('High-Res', 85.25),
    ('DenseNet Original', 89.51),
    ('DenseNet Improved', 89.84),
    ('3-Model Ensemble', 90.16),
    ('5-Fold K-Fold Ensemble', 92.46)
]
models = [m[0] for m in model_progression]
accs = [m[1] for m in model_progression]
colors_prog = ['#95a5a6', '#95a5a6', '#7f8c8d', '#3498db', '#2980b9', '#27ae60', '#e74c3c']

ax4.plot(range(len(models)), accs, marker='o', markersize=12, linewidth=3, 
         color='#2c3e50', label='Model Evolution')
for i, (model, acc, color) in enumerate(zip(models, accs, colors_prog)):
    ax4.scatter(i, acc, s=300, color=color, edgecolor='black', linewidth=2, zorder=5)
    ax4.text(i, acc + 1.2, f'{acc:.2f}%', ha='center', fontweight='bold', fontsize=10)

ax4.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='Target 90%', alpha=0.7)
ax4.axhline(y=94, color='green', linestyle='--', linewidth=2, label='Target 94%', alpha=0.7)
ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
ax4.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Model Performance Evolution', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right', fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([75, 95])

# Fill area under curve
ax4.fill_between(range(len(models)), accs, alpha=0.2, color='#3498db')

# 5. Error Distribution
ax5 = fig.add_subplot(gs[2, 0])
categories = ['True Negatives\n(Correct Cardio)', 'False Positives\n(Wrong Pneumo)', 
              'False Negatives\n(Missed Pneumo)', 'True Positives\n(Correct Pneumo)']
values = [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
colors_err = ['#2ecc71', '#e67e22', '#e74c3c', '#3498db']
explode = (0, 0.1, 0.1, 0)

wedges, texts, autotexts = ax5.pie(values, labels=categories, autopct='%1.1f%%',
                                    colors=colors_err, explode=explode,
                                    startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax5.set_title('Prediction Distribution', fontsize=14, fontweight='bold')

# 6. Sensitivity & Specificity
ax6 = fig.add_subplot(gs[2, 1])
sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])  # True Positive Rate
specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # True Negative Rate
ppv = results['precision']  # Positive Predictive Value
npv = cm[0,0] / (cm[0,0] + cm[1,0])  # Negative Predictive Value

metrics_med = ['Sensitivity\n(Recall)', 'Specificity', 'PPV\n(Precision)', 'NPV']
values_med = [sensitivity*100, specificity*100, ppv*100, npv*100]
colors_med = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

bars = ax6.bar(metrics_med, values_med, color=colors_med, edgecolor='black', linewidth=1.5)
ax6.axhline(y=90, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax6.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax6.set_title('Clinical Performance Metrics', fontsize=14, fontweight='bold')
ax6.set_ylim([0, 100])
ax6.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 7. Key Statistics Summary
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

summary_text = f"""
╔══════════════════════════════════════╗
║   K-FOLD ENSEMBLE FINAL RESULTS      ║
╚══════════════════════════════════════╝

🎯 ACCURACY: {results['accuracy']*100:.2f}%
   Gap to 94%: {(0.94-results['accuracy'])*100:.2f}%

📊 PERFORMANCE METRICS:
   Precision:  {results['precision']*100:.2f}%
   Recall:     {results['recall']*100:.2f}%
   F1-Score:   {results['f1']*100:.2f}%
   AUC-ROC:    {results['auc']*100:.2f}%

🔬 CLINICAL METRICS:
   Sensitivity: {sensitivity*100:.2f}%
   Specificity: {specificity*100:.2f}%
   PPV:         {ppv*100:.2f}%
   NPV:         {npv*100:.2f}%

📈 ENSEMBLE DETAILS:
   Number of Models: 5
   Average Fold Acc: {np.mean(results['fold_accs'])*100:.2f}%
   Best Fold:        {max(results['fold_accs'])*100:.2f}%
   Threshold:        {results['threshold']:.2f}

✅ IMPROVEMENT FROM BASELINE:
   SimpleCNN:     +{(results['accuracy']-0.825)*100:.2f}%
   DenseNet Orig: +{(results['accuracy']-0.8951)*100:.2f}%
   3-Model Ens:   +{(results['accuracy']-0.9016)*100:.2f}%

🎊 STATUS: 92%+ ACHIEVED!
   (Excellent for medical AI screening)
"""

ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Save figure
output_path = RESULTS_DIR / 'kfold_ensemble_results.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Visualization saved: {output_path}")
plt.close()

print("\n🎉 K-Fold Ensemble Results Visualization Created!")
print(f"📊 Final Accuracy: {results['accuracy']*100:.2f}%")
print(f"🎯 Gap to 94%: {(0.94-results['accuracy'])*100:.2f}%")
