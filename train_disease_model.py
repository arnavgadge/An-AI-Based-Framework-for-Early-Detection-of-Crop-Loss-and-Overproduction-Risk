"""
train_disease_model.py
═══════════════════════════════════════════════════════════════════════════════
Trains a MobileNetV2 CNN on the onion disease image dataset.
Saves:  onion_disease_model.h5
        class_indices.pkl
Both files go into the SAME folder as this script (your project root).

DATASET EXPECTED AT:
    <project_root>/archive/onion datasets/<ClassName>/image.jpg ...

CLASSES DETECTED FROM YOUR FOLDER (from screenshot):
    Alternaria_D, Botrytis Leaf Blight, Bulb Rot, Bulb_blight-D,
    Caterpillar-P, Downy mildew, Fusarium-D, Healthy leaves,
    Iris yellow virus_augment, onion1, Purple blotch, Rust,
    stemphylium Leaf Blight, Virosis-D, Xanthomonas Leaf Blight

HOW TO RUN:
    pip install tensorflow pillow numpy scikit-learn
    python train_disease_model.py

AFTER TRAINING:
    Copy  onion_disease_model.h5  and  class_indices.pkl
    into your project root (same folder as cv.py).
    Then the 🧠 AI Disease Detection tab in cv.py will work.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import pickle
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Locate dataset ────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "archive", "onion datasets")

if not os.path.isdir(DATASET_DIR):
    # Try alternate location — sometimes just "onion datasets" in root
    alt = os.path.join(SCRIPT_DIR, "onion datasets")
    if os.path.isdir(alt):
        DATASET_DIR = alt
    else:
        print("❌ Dataset folder not found!")
        print(f"   Tried: {DATASET_DIR}")
        print(f"   Tried: {alt}")
        print("\n   Make sure the folder structure is:")
        print("   <project>/archive/onion datasets/<ClassName>/image.jpg")
        sys.exit(1)

MODEL_SAVE_PATH  = os.path.join(SCRIPT_DIR, "onion_disease_model.h5")
CLASS_IDX_PATH   = os.path.join(SCRIPT_DIR, "class_indices.pkl")
HISTORY_PATH     = os.path.join(SCRIPT_DIR, "training_history.json")

print("=" * 65)
print("  ONION DISEASE CNN TRAINING")
print("=" * 65)
print(f"  Dataset : {DATASET_DIR}")
print(f"  Model   : {MODEL_SAVE_PATH}")
print(f"  Started : {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
print("=" * 65)

# ── Scan classes ──────────────────────────────────────────────────────────────
all_folders = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

if not all_folders:
    print("❌ No sub-folders found in dataset directory.")
    sys.exit(1)

print(f"\n📁 Detected {len(all_folders)} class folders:")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
class_image_counts = {}
for folder in all_folders:
    folder_path = os.path.join(DATASET_DIR, folder)
    count = sum(
        1 for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in VALID_EXTS
    )
    class_image_counts[folder] = count
    status = "✅" if count > 0 else "⚠️  EMPTY"
    print(f"   {status}  {folder:<35}  {count} images")

# Remove empty folders from training
valid_classes = [c for c, cnt in class_image_counts.items() if cnt > 0]
if len(valid_classes) < 2:
    print("\n❌ Need at least 2 non-empty class folders to train.")
    sys.exit(1)

total_images = sum(class_image_counts[c] for c in valid_classes)
print(f"\n   Total images for training: {total_images}")
print(f"   Classes with images      : {len(valid_classes)}")

# ── Import TensorFlow ─────────────────────────────────────────────────────────
print("\n⏳ Importing TensorFlow...")
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    print(f"✅ TensorFlow {tf.__version__} loaded")
except ImportError:
    print("❌ TensorFlow not installed.")
    print("   Run:  pip install tensorflow")
    sys.exit(1)

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS_HEAD = 15       # Phase 1: train head only (fast)
EPOCHS_FINE = 10       # Phase 2: fine-tune top layers (slower, optional)
VAL_SPLIT   = 0.20
SEED        = 42

print(f"\n⚙️  Config:")
print(f"   Image size  : {IMG_SIZE}")
print(f"   Batch size  : {BATCH_SIZE}")
print(f"   Val split   : {int(VAL_SPLIT*100)}%")
print(f"   Phase 1 epochs (head training): {EPOCHS_HEAD}")
print(f"   Phase 2 epochs (fine-tuning)  : {EPOCHS_FINE}")

# ── Data generators ───────────────────────────────────────────────────────────
print("\n📦 Creating data generators...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VAL_SPLIT,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VAL_SPLIT
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

NUM_CLASSES    = train_gen.num_classes
CLASS_INDICES  = train_gen.class_indices   # {class_name: index}

print(f"\n✅ Training generator  : {train_gen.samples} images, {NUM_CLASSES} classes")
print(f"✅ Validation generator: {val_gen.samples} images")
print(f"\n🗂️  Class → Index mapping:")
for cls_name, idx in sorted(CLASS_INDICES.items(), key=lambda x: x[1]):
    print(f"   [{idx:02d}]  {cls_name}")

# ── Build model ───────────────────────────────────────────────────────────────
print("\n🔨 Building MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"     # downloads ~14MB pretrained weights
)
base_model.trainable = False   # freeze for Phase 1

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
], name="onion_disease_detector")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"✅ Model built. Trainable parameters: {model.count_params():,}")
model.summary(line_length=80, print_fn=lambda x: print(f"   {x}"))

# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_path = os.path.join(SCRIPT_DIR, "best_disease_model.h5")

callbacks_phase1 = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ── PHASE 1: Train head only ──────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  PHASE 1: Training classifier head (base frozen)")
print("─" * 65)

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    callbacks=callbacks_phase1,
    verbose=1
)

best_phase1_acc = max(history1.history.get("val_accuracy", [0]))
print(f"\n✅ Phase 1 complete. Best val_accuracy: {best_phase1_acc:.4f}")

# ── PHASE 2: Fine-tune top layers of base ────────────────────────────────────
print("\n" + "─" * 65)
print("  PHASE 2: Fine-tuning top 30 layers of MobileNetV2")
print("─" * 65)

# Unfreeze top 30 layers of the base model
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    callbacks=callbacks_phase2,
    verbose=1
)

best_phase2_acc = max(history2.history.get("val_accuracy", [0]))
print(f"\n✅ Phase 2 complete. Best val_accuracy: {best_phase2_acc:.4f}")

# ── Save final model ──────────────────────────────────────────────────────────
print("\n💾 Saving model...")

# Back up old model if it exists
if os.path.exists(MODEL_SAVE_PATH):
    backup = MODEL_SAVE_PATH.replace(".h5", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    shutil.copy(MODEL_SAVE_PATH, backup)
    print(f"   Old model backed up → {backup}")

model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved         → {MODEL_SAVE_PATH}")

# Save class index mapping
with open(CLASS_IDX_PATH, "wb") as f:
    pickle.dump(CLASS_INDICES, f)
print(f"✅ Class indices saved → {CLASS_IDX_PATH}")

# Save training history
all_history = {
    "phase1_accuracy":     history1.history.get("accuracy", []),
    "phase1_val_accuracy": history1.history.get("val_accuracy", []),
    "phase1_loss":         history1.history.get("loss", []),
    "phase1_val_loss":     history1.history.get("val_loss", []),
    "phase2_accuracy":     history2.history.get("accuracy", []),
    "phase2_val_accuracy": history2.history.get("val_accuracy", []),
    "phase2_loss":         history2.history.get("loss", []),
    "phase2_val_loss":     history2.history.get("val_loss", []),
}
with open(HISTORY_PATH, "w") as f:
    json.dump(all_history, f, indent=2)
print(f"✅ Training history    → {HISTORY_PATH}")

# ── Final summary ─────────────────────────────────────────────────────────────
final_val_acc = max(best_phase1_acc, best_phase2_acc)

print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print("=" * 65)
print(f"  Dataset           : {total_images} images, {NUM_CLASSES} classes")
print(f"  Best val_accuracy : {final_val_acc * 100:.2f}%")
print(f"  Model file        : {MODEL_SAVE_PATH}")
print(f"  Class map file    : {CLASS_IDX_PATH}")
print(f"  Finished          : {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
print("=" * 65)
print()
print("  ✅ NEXT STEPS:")
print("  1. Make sure onion_disease_model.h5 and class_indices.pkl")
print("     are in the same folder as cv.py")
print("  2. Restart your Streamlit app:")
print("     streamlit run cv.py")
print("  3. Go to Admin → 🧠 AI Disease Detection tab")
print("=" * 65)
