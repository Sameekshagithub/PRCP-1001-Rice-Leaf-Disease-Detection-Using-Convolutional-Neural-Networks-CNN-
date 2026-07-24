"""
train_model.py

Builds and trains a Convolutional Neural Network COMPLETELY FROM SCRATCH
(random weight initialization) for Rice Leaf Disease classification.

No pretrained / transfer-learning model is used here (no VGG16, no MobileNetV2,
no ImageNet weights) - every layer is defined and trained from zero on your
own dataset.

--------------------------------------------------------------------------
BEFORE RUNNING THIS SCRIPT:
Organize your dataset into this structure:

    dataset/
    ├── Bacterial Leaf Blight/
    │   ├── img1.jpg
    │   └── ...
    ├── Brown Spot/
    │   ├── img1.jpg
    │   └── ...
    └── Leaf Smut/
        ├── img1.jpg
        └── ...

Then run:
    python train_model.py
--------------------------------------------------------------------------
"""

import os
import json
import numpy as np
import splitfolders

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 40

DATASET_DIR = "dataset"          # raw dataset, organized as one folder per class
SPLIT_DIR = "dataset_split"      # created automatically: train / val / test
MODEL_OUTPUT = "Rice_Leaf_Disease_Scratch_Model.keras"
CLASS_NAMES_FILE = "class_names.json"

np.random.seed(SEED)

# ---------------------------------------------------------------
# 1. Split dataset into train / val / test (70 / 15 / 15)
# ---------------------------------------------------------------
print("Splitting dataset into train/val/test ...")
splitfolders.ratio(
    input=DATASET_DIR,
    output=SPLIT_DIR,
    seed=SEED,
    ratio=(0.70, 0.15, 0.15)
)

train_dir = os.path.join(SPLIT_DIR, "train")
val_dir = os.path.join(SPLIT_DIR, "val")
test_dir = os.path.join(SPLIT_DIR, "test")

# ---------------------------------------------------------------
# 2. Data augmentation (training only) + rescaling
# ---------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True, seed=SEED
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

# Save class name mapping so app.py always uses the correct label order
class_indices = train_generator.class_indices
class_names = [cls for cls, idx in sorted(class_indices.items(), key=lambda x: x[1])]
with open(CLASS_NAMES_FILE, "w") as f:
    json.dump(class_names, f)

print("Class label mapping:", class_indices)

# ---------------------------------------------------------------
# 3. Build the CNN from scratch (no pretrained weights, no transfer learning)
# ---------------------------------------------------------------
model = Sequential(name="Rice_Leaf_CNN_From_Scratch")

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(len(class_names), activation="softmax"))

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_OUTPUT, monitor="val_accuracy", save_best_only=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# ---------------------------------------------------------------
# 5. Evaluate on the held-out test set
# ---------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Final Test Loss    : {test_loss:.4f}")

# Model is already saved by ModelCheckpoint as the best epoch,
# but save again to guarantee the final file exists even if training was very short.
model.save(MODEL_OUTPUT)
print(f"\nModel saved to: {MODEL_OUTPUT}")
print(f"Class names saved to: {CLASS_NAMES_FILE}")
print("\nYou can now run the Flask app with: python app.py")
