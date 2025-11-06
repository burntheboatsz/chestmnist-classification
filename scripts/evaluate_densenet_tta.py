# evaluate_densenet_tta.py
"""
Test-Time Augmentation (TTA) for DenseNet121
Applies multiple augmentations during inference and averages predictions
Expected improvement: +1-2% accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Define paths relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAINED_MODELS_DIR = PROJECT_DIR / 'trained_models'
RESULTS_DIR = PROJECT_DIR / 'results'

from torchvision import transforms
from data.datareader_highres import get_data_loaders
from models.model_densenet import get_densenet_model
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = TRAINED_MODELS_DIR / 'best_densenet_model.pth'
BATCH_SIZE = 8  # Smaller because TTA multiplies samples
N_TTA = 5  # Number of augmented versions per image

print("=" * 70)
print("DENSENET121 EVALUATION WITH TEST-TIME AUGMENTATION (TTA)")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"TTA Augmentations: {N_TTA}")
print("=" * 70)

# Load model
print("\n🔧 Loading DenseNet121 model...")
model = get_densenet_model(
    model_type='standard',
    num_classes=1,
    dropout_rate=0.5,
    freeze_backbone=False
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✓ Model loaded")

# Load data (without augmentation for baseline)
print("\n📊 Loading validation dataset...")
_, val_loader, _, _ = get_data_loaders(
    batch_size=BATCH_SIZE,
    use_augmentation=False
)
print(f"✓ Validation samples: {len(val_loader.dataset)}")

# Define TTA transforms
tta_transforms = [
    # Original
    transforms.Compose([]),
    
    # Rotation variants
    transforms.Compose([
        transforms.RandomRotation(degrees=5)
    ]),
    
    # Affine variants
    transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
    ]),
    
    # Brightness/Contrast
    transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ]),
    
    # Combined
    transforms.Compose([
        transforms.RandomRotation(degrees=3),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
    ]),
]

def apply_tta(image, model, transforms_list, device):
    """
    Apply Test-Time Augmentation
    
    Args:
        image: Input image tensor [1, C, H, W]
        model: PyTorch model
        transforms_list: List of transform compositions
        device: torch device
    
    Returns:
        Averaged prediction probability
    """
    predictions = []
    
    with torch.no_grad():
        for transform in transforms_list:
            # Apply transform
            if len(list(transform.transforms)) > 0:
                # Convert to PIL, apply transform, back to tensor
                img_pil = transforms.ToPILImage()(image.squeeze(0).cpu())
                img_transformed = transform(img_pil)
                img_tensor = transforms.ToTensor()(img_transformed).unsqueeze(0).to(device)
            else:
                img_tensor = image
            
            # Get prediction
            output = model(img_tensor)
            prob = torch.sigmoid(output).cpu().item()
            predictions.append(prob)
    
    # Average predictions
    avg_prob = np.mean(predictions)
    return avg_prob

# Evaluate with TTA
print("\n🔍 Evaluating with Test-Time Augmentation...")
print(f"  Each image will be augmented {N_TTA} times")
print(f"  Predictions will be averaged for final decision")
print()

all_labels = []
all_preds_tta = []
all_probs_tta = []
all_preds_single = []  # For comparison

from tqdm import tqdm

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="TTA Evaluation"):
        images = images.to(DEVICE)
        labels_np = labels.cpu().numpy()
        
        # Process each image individually for TTA
        for i in range(images.size(0)):
            image = images[i:i+1]
            label = labels_np[i]
            
            # TTA prediction
            prob_tta = apply_tta(image, model, tta_transforms[:N_TTA], DEVICE)
            pred_tta = 1 if prob_tta > 0.5 else 0
            
            # Single prediction (no augmentation) for comparison
            output_single = model(image)
            prob_single = torch.sigmoid(output_single).cpu().item()
            pred_single = 1 if prob_single > 0.5 else 0
            
            all_labels.append(label)
            all_probs_tta.append(prob_tta)
            all_preds_tta.append(pred_tta)
            all_preds_single.append(pred_single)

all_labels = np.array(all_labels)
all_preds_tta = np.array(all_preds_tta)
all_probs_tta = np.array(all_probs_tta)
all_preds_single = np.array(all_preds_single)

# Calculate metrics
acc_tta = accuracy_score(all_labels, all_preds_tta)
acc_single = accuracy_score(all_labels, all_preds_single)
f1_tta = f1_score(all_labels, all_preds_tta)
f1_single = f1_score(all_labels, all_preds_single)
auc_tta = roc_auc_score(all_labels, all_probs_tta)

cm_tta = confusion_matrix(all_labels, all_preds_tta)

# Results
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print(f"\n📊 Single Prediction (No TTA):")
print(f"  Accuracy:  {acc_single:.4f} ({acc_single*100:.2f}%)")
print(f"  F1-Score:  {f1_single:.4f} ({f1_single*100:.2f}%)")

print(f"\n🎯 With Test-Time Augmentation ({N_TTA} augmentations):")
print(f"  Accuracy:  {acc_tta:.4f} ({acc_tta*100:.2f}%)")
print(f"  F1-Score:  {f1_tta:.4f} ({f1_tta*100:.2f}%)")
print(f"  AUC-ROC:   {auc_tta:.4f} ({auc_tta*100:.2f}%)")

print(f"\n📈 Improvement:")
acc_gain = (acc_tta - acc_single) * 100
f1_gain = (f1_tta - f1_single) * 100
print(f"  Accuracy:  {acc_gain:+.2f}%")
print(f"  F1-Score:  {f1_gain:+.2f}%")

print(f"\n📈 Confusion Matrix (TTA):")
print(f"                 Predicted")
print(f"               Cardio  Pneumo")
print(f"Actual Cardio    {cm_tta[0,0]:3d}     {cm_tta[0,1]:3d}")
print(f"       Pneumo    {cm_tta[1,0]:3d}     {cm_tta[1,1]:3d}")

if acc_tta >= 0.95:
    print(f"\n🎉 TARGET 95% REACHED! Accuracy: {acc_tta*100:.2f}%")
elif acc_tta > acc_single:
    print(f"\n✅ TTA IMPROVED performance by {acc_gain:.2f}%")
else:
    print(f"\n⚠️ TTA did not improve performance")

print("\n" + "=" * 70)
print("✅ TTA Evaluation completed!")
print("=" * 70)
