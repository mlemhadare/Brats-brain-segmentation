import sys
sys.path.append('src/')
try:
    import display
except ModuleNotFoundError:
    sys.path.append('../src/')
    import display

import data_process as dp
import numpy as np
import nibabel as nib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

"""
Comprehensive Brain Tumor Segmentation Demo
============================================
This demo showcases the complete pipeline from raw 3D brain MRI to predicted tumor segmentation.

Pipeline Steps:
1. Load raw .nii.gz brain MRI data
2. Preprocess and scale the images
3. Run prediction using trained .keras model
4. Visualize results with optional ground truth comparison

Usage:
    - Update 'patient' variable with your patient ID
    - Update 'model_path' with your .keras model path
    - Set 'has_ground_truth' to True if ground truth segmentation is available
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

print("=" * 70)
print("BRAIN TUMOR SEGMENTATION - PATIENT SELECTION")
print("=" * 70)

# Dataset selection
print("\nSelect Dataset:")
print("  1. Training Data (with Ground Truth segmentation)")
print("  2. Validation Data (no Ground Truth available)")

while True:
    try:
        dataset_choice = input("\nEnter choice (1 or 2): ").strip()
        
        if dataset_choice == "1":
            data_type = "training"
            has_ground_truth = True
            print("✓ Selected: Training Data (with Ground Truth)")
            break
        elif dataset_choice == "2":
            data_type = "validation"
            has_ground_truth = False
            print("✓ Selected: Validation Data (no Ground Truth)")
            break
        else:
            print("✗ Error: Please enter 1 or 2.")
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
        sys.exit(0)

# Interactive patient selection
while True:
    try:
        if data_type == "training":
            patient_input = input("\nEnter Patient ID (1-369): ").strip()
            patient_num = int(patient_input)
            
            if 1 <= patient_num <= 369:
                # Format with leading zeros (e.g., 001, 002, ..., 369)
                patient = f"{patient_num:03d}"
                print(f"✓ Selected Patient ID: {patient}")
                break
            else:
                print("✗ Error: Training Patient ID must be between 1 and 369. Please try again.")
        else:  # validation
            patient_input = input("\nEnter Patient ID (1-125): ").strip()
            patient_num = int(patient_input)
            
            if 1 <= patient_num <= 125:
                # Format with leading zeros (e.g., 001, 002, ..., 125)
                patient = f"{patient_num:03d}"
                print(f"✓ Selected Patient ID: {patient}")
                break
            else:
                print("✗ Error: Validation Patient ID must be between 1 and 125. Please try again.")
    except ValueError:
        print("✗ Error: Please enter a valid number.")
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
        sys.exit(0)

# Model path - update to your .keras model
model_path = "../model/brats_3d_simple_unet_2d0aede9dd9145ca85577b7004621d5f_epoch98.keras"

# Data paths
if data_type == "training":
    data_dir = "../data/BraTS2020_TrainingData"
    patient_folder = f"BraTS20_Training_{patient}"
    patient_path = os.path.join(data_dir, patient_folder, f"BraTS20_Training_{patient}_")
else:  # validation
    data_dir = "../data/BraTS2020_ValidationData"
    patient_folder = f"BraTS20_Validation_{patient}"
    patient_path = os.path.join(data_dir, patient_folder, f"BraTS20_Validation_{patient}_")

# ============================================================================
# MODEL SETUP
# ============================================================================

print("=" * 70)
print("BRAIN TUMOR SEGMENTATION DEMO")
print("=" * 70)
print(f"\nDataset: {data_type.upper()}")
print(f"Patient ID: {patient}")
print(f"Model: {model_path}")
print(f"Ground Truth Available: {has_ground_truth}")
print("\n" + "=" * 70)

# Initialize scaler
scaler = MinMaxScaler()

# Loss function configuration (required for loading model)
wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# ============================================================================
# STEP 1: LOAD RAW MRI DATA
# ============================================================================

print("\n[STEP 1/5] Loading raw MRI data...")

# Load the three sequences: FLAIR, T1CE, T2
try:
    image_flair_raw = nib.load(patient_path + "flair.nii").get_fdata()
    image_t1ce_raw = nib.load(patient_path + "t1ce.nii").get_fdata()
    image_t2_raw = nib.load(patient_path + "t2.nii").get_fdata()
    print(f"  ✓ Loaded FLAIR: {image_flair_raw.shape}")
    print(f"  ✓ Loaded T1CE:  {image_t1ce_raw.shape}")
    print(f"  ✓ Loaded T2:    {image_t2_raw.shape}")
except FileNotFoundError as e:
    print(f"  ✗ Error loading patient data: {e}")
    print("  Please check patient ID and data directory paths.")
    sys.exit(1)

# Load ground truth if available
if has_ground_truth:
    try:
        ground_truth = nib.load(patient_path + "seg.nii").get_fdata()
        print(f"  ✓ Loaded Ground Truth: {ground_truth.shape}")
    except FileNotFoundError:
        print("  ⚠ Ground truth not found, continuing without comparison")
        has_ground_truth = False

# ============================================================================
# STEP 2: DISPLAY ORIGINAL DATA
# ============================================================================

print("\n[STEP 2/5] Displaying original T1CE brain scan...")
if has_ground_truth:
    print("  📊 WINDOW: Original Brain MRI + Ground Truth Segmentation")
else:
    print("  📊 WINDOW: Original Brain MRI")
print("  (Close the window to continue)")

if has_ground_truth:
    display.display3DCuts(image_t1ce_raw, ground_truth, seg_color_map='jet', seg_alpha=0.5, 
                          window_title='STEP 2: Original Brain MRI + Ground Truth')
else:
    display.display3DCuts(image_t1ce_raw, window_title='STEP 2: Original Brain MRI')

# ============================================================================
# STEP 3: PREPROCESS DATA
# ============================================================================

print("\n[STEP 3/5] Preprocessing MRI data...")

# Scale each sequence to [0, 1] range
image_flair = scaler.fit_transform(image_flair_raw.reshape(-1, image_flair_raw.shape[-1])).reshape(image_flair_raw.shape)
image_t1ce = scaler.fit_transform(image_t1ce_raw.reshape(-1, image_t1ce_raw.shape[-1])).reshape(image_t1ce_raw.shape)
image_t2 = scaler.fit_transform(image_t2_raw.reshape(-1, image_t2_raw.shape[-1])).reshape(image_t2_raw.shape)

# Stack the 3 sequences into a single 4D array
processed_image = np.stack([image_flair, image_t1ce, image_t2], axis=3)
print(f"  ✓ Stacked sequences: {processed_image.shape}")

# Crop to 128x128x128 (model input size)
processed_image = processed_image[56:184, 56:184, 13:141]
print(f"  ✓ Cropped to model input size: {processed_image.shape}")

# Display processed image
print("  ✓ Displaying preprocessed T1CE scan...")
print("  📊 WINDOW: Preprocessed Brain (Cropped to 128×128×128)")
print("  (Close the window to continue)")
display.display3DCuts(processed_image[:, :, :, 1], window_title='STEP 3: Preprocessed Brain (128×128×128)')

# ============================================================================
# STEP 4: RUN PREDICTION
# ============================================================================

print("\n[STEP 4/5] Loading model and running prediction...")

# Load the trained model
try:
    model = load_model(
        model_path,
        custom_objects={
            'dice_loss_plus_1focal_loss': total_loss,
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'f1-score': sm.metrics.FScore(),
            'DiceLoss': sm.losses.DiceLoss,
            'CategoricalFocalLoss': sm.losses.CategoricalFocalLoss,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss,
            'total_loss': total_loss
        },
        compile=False
    )
    # Recompile with the loss function
    model.compile(
        optimizer='adam',
        loss=total_loss,
        metrics=[sm.metrics.FScore(), sm.metrics.IOUScore(threshold=0.5)]
    )
    print(f"  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    sys.exit(1)

# Prepare input (add batch dimension)
input_image = np.expand_dims(processed_image, axis=0)
print(f"  ✓ Input shape: {input_image.shape}")

# Run prediction
print("  ⚙ Running prediction...")
prediction = model.predict(input_image, verbose=0)
print(f"  ✓ Prediction complete: {prediction.shape}")

# Post-process prediction
# Get the class with highest probability for each voxel
prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
print(f"  ✓ Prediction argmax: {prediction_argmax.shape}")

# Expand prediction back to original image size (240x240x155)
result = np.zeros([240, 240, 155])
result[56:184, 56:184, 13:141] = prediction_argmax

# Convert label 3 back to label 4 (edema)
result[result == 3] = 4

print(f"  ✓ Final prediction: {result.shape}")
print(f"  ✓ Unique labels: {np.unique(result)}")

# ============================================================================
# STEP 5: VISUALIZE RESULTS
# ============================================================================

print("\n[STEP 5/5] Visualizing results...")

if has_ground_truth:
    print("  📊 FINAL WINDOW: Ground Truth vs Model Prediction Comparison")
    print("     ├─ TOP ROW:    Ground Truth Segmentation")
    print("     └─ BOTTOM ROW: Model Prediction")
    print("  (Close the window to exit)")
    
    # Display side-by-side comparison
    display.displayPred3DCuts2(image_t1ce_raw, ground_truth, result, 
                               window_title='STEP 5: Ground Truth vs Model Prediction')
else:
    print("  📊 FINAL WINDOW: Original Brain vs Model Prediction")
    print("     ├─ TOP ROW:    Original Brain (for manual verification)")
    print("     └─ BOTTOM ROW: Model Prediction")
    print("  (Close the window to exit)")
    
    # Create a "dummy" ground truth (zeros) to use the comparison display
    # This shows original brain on top, prediction on bottom for manual verification
    dummy_ground_truth = np.zeros_like(result)
    display.displayPred3DCuts2(image_t1ce_raw, dummy_ground_truth, result, 
                               window_title='STEP 5: Original Brain vs Model Prediction')

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nTumor Segmentation Summary:")
print(f"  - Background (0):        {np.sum(result == 0):,} voxels")
print(f"  - Necrotic core (1):     {np.sum(result == 1):,} voxels")
print(f"  - Enhancing tumor (2):   {np.sum(result == 2):,} voxels")
print(f"  - Edema (4):             {np.sum(result == 4):,} voxels")

total_tumor_voxels = np.sum((result == 1) | (result == 2) | (result == 4))
print(f"\n  Total tumor voxels:      {total_tumor_voxels:,}")
print(f"  Tumor volume:            ~{total_tumor_voxels} mm³")

if has_ground_truth:
    # Calculate simple accuracy metrics
    gt_tumor = (ground_truth > 0)
    pred_tumor = (result > 0)
    
    intersection = np.sum(gt_tumor & pred_tumor)
    union = np.sum(gt_tumor | pred_tumor)
    
    if union > 0:
        iou = intersection / union
        print(f"\n  Tumor IoU (Dice-like):   {iou:.4f}")

print("\n" + "=" * 70)
