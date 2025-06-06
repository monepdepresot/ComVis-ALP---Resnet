import os
import math
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.io import loadmat

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set seed for reproducibility
tf.random.set_seed(1)

# LIVE dataset configuration
BASE_DIR = 'databaserelease2/'
distortion_folders = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]

def parse_info_files(base_dir, distortion_folders):
    """Parse info.txt files to create filename to index mapping"""
    mapping, idx = {}, 0
    for folder in distortion_folders:
        folder_map = {}
        info_path = os.path.join(base_dir, folder, "info.txt")
        if os.path.exists(info_path):
            with open(info_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        folder_map[parts[1]] = idx
                        idx += 1
        mapping[folder] = folder_map
    return mapping

# Load DMOS scores
data = loadmat(BASE_DIR + "dmos.mat")
dmos_original = data["dmos"].flatten()
orgs = data["orgs"].flatten()
dmos_map = parse_info_files(BASE_DIR, distortion_folders)

# Create organized folder structure for train/val/test splits
if not os.path.isdir(BASE_DIR + 'train/'):
    os.makedirs(BASE_DIR + 'train/distorted')
    os.makedirs(BASE_DIR + 'val/distorted') 
    os.makedirs(BASE_DIR + 'test/distorted')

# Create CSV files to store image paths and DMOS scores for each split
train_data = []
val_data = []
test_data = []

# Move image files and collect DMOS scores
for folder in distortion_folders:
    folder_path = os.path.join(BASE_DIR, folder)
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.bmp') and f.startswith('img')]
        
        # Filter files that have valid DMOS scores and are distorted images
        valid_files = []
        valid_dmos = []
        for fname in files:
            i = dmos_map[folder].get(fname)
            if i is not None and i < len(orgs) and orgs[i] == 0:  # Only distorted images
                valid_files.append(fname)
                valid_dmos.append(dmos_original[i])
        
        number_of_images = len(valid_files)
        if number_of_images == 0:
            continue
            
        n_train = int((number_of_images * 0.6) + 0.5)
        n_valid = int((number_of_images * 0.25) + 0.5)
        n_test = number_of_images - n_train - n_valid
        print(f"{folder}: {number_of_images} images -> train: {n_train}, val: {n_valid}, test: {n_test}")
        
        for idx, (fname, dmos_score) in enumerate(zip(valid_files, valid_dmos)):
            src_path = os.path.join(folder_path, fname)
            
            if idx < n_train:
                dst_path = os.path.join(BASE_DIR, "train/distorted", f"{folder}_{fname}")
                train_data.append([dst_path, dmos_score])
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
            elif idx < n_train + n_valid:
                dst_path = os.path.join(BASE_DIR, "val/distorted", f"{folder}_{fname}")
                val_data.append([dst_path, dmos_score])
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
            else:
                dst_path = os.path.join(BASE_DIR, "test/distorted", f"{folder}_{fname}")
                test_data.append([dst_path, dmos_score])
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

# Save the data splits with DMOS scores
pd.DataFrame(train_data, columns=['filepath', 'dmos']).to_csv(BASE_DIR + 'train_data.csv', index=False)
pd.DataFrame(val_data, columns=['filepath', 'dmos']).to_csv(BASE_DIR + 'val_data.csv', index=False)  
pd.DataFrame(test_data, columns=['filepath', 'dmos']).to_csv(BASE_DIR + 'test_data.csv', index=False)

print(f"Total train samples: {len(train_data)}")
print(f"Total val samples: {len(val_data)}")
print(f"Total test samples: {len(test_data)}")
print(f"Total images used: {len(train_data) + len(val_data) + len(test_data)}")
print("="*50)

# Custom data generator for regression that reads DMOS scores from CSV
class DMOSImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_file, target_size=(224, 224), batch_size=4, 
                 preprocessing_function=None, shuffle=True):
        self.data = pd.read_csv(csv_file)
        # Remove DMOS normalization - use raw values
        self.target_size = target_size
        self.batch_size = batch_size  
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = []
        batch_dmos = []
        
        for i in batch_indices:
            # Load image
            img_path = self.data.iloc[i]['filepath'] 
            img = keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            
            if self.preprocessing_function:
                img_array = self.preprocessing_function(img_array)
            else:
                img_array = img_array / 255.0  # Default normalization
                
            batch_images.append(img_array)
            batch_dmos.append(self.data.iloc[i]['dmos'])
        
        return np.array(batch_images), np.array(batch_dmos, dtype=np.float32)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Create data generators similar to the original tutorial
train_gen = DMOSImageDataGenerator(
    BASE_DIR + 'train_data.csv',
    target_size=(224, 224),
    batch_size=4,
    shuffle=True
)

val_gen = DMOSImageDataGenerator(
    BASE_DIR + 'val_data.csv', 
    target_size=(224, 224),
    batch_size=4,
    shuffle=False
)

test_gen = DMOSImageDataGenerator(
    BASE_DIR + 'test_data.csv',
    target_size=(224, 224), 
    batch_size=4,
    shuffle=False
)

# Test the generators
train_batch = train_gen[0]
print("Train batch shape:", train_batch[0].shape)
print("Train DMOS scores:", train_batch[1])

test_batch = test_gen[0]
print("Test batch shape:", test_batch[0].shape) 
print("Test DMOS scores:", test_batch[1])

def show(batch, pred_dmos=None):
    plt.figure(figsize=(10,10))
    for i in range(min(4, len(batch[0]))):
        plt.subplot(2,2,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i])
        
        lbl = f"DMOS: {batch[1][i]:.2f}"
        if pred_dmos is not None:
            lbl += f" / Pred: {pred_dmos[i]:.2f}"
        plt.xlabel(lbl)
    plt.show()

show(test_batch)

print("\nInitializing ResNet50 Transfer Learning Model")
print("="*50)

# Load pretrained ResNet50 with weights and correct input shape
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,  # Don't include the classification layers
    input_shape=(224, 224, 3),
    pooling='avg'  # Use average pooling at the end
)
print("Base ResNet50 architecture:")
base_model.summary()

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = True
print("\nFrozen layers (trainable parameters):")

# Create new model with our regression head
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
outputs = layers.Dense(1)(x)  # Single output for regression
model = keras.Model(inputs, outputs)

print("\nFinal architecture with regression head:")
model.summary()

# Compile for regression with RMSE metric
loss = keras.losses.MeanSquaredError()
# Use constant learning rate instead of schedule
initial_learning_rate = 0.001
optim = keras.optimizers.Adam(
    learning_rate=initial_learning_rate,
    weight_decay=1e-5  # Add weight decay
)
metrics = ["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# Get preprocessing function for ResNet50
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# Create generators with ResNet50 preprocessing and larger batch size
train_gen = DMOSImageDataGenerator(
    BASE_DIR + 'train_data.csv',
    target_size=(224, 224),
    batch_size=64,  # Increased from 32
    preprocessing_function=preprocess_input,
    shuffle=True
)

val_gen = DMOSImageDataGenerator(
    BASE_DIR + 'val_data.csv',
    target_size=(224, 224), 
    batch_size=64,  # Increased from 32
    preprocessing_function=preprocess_input,
    shuffle=False
)

test_gen = DMOSImageDataGenerator(
    BASE_DIR + 'test_data.csv',
    target_size=(224, 224),
    batch_size=64,  # Increased from 32
    preprocessing_function=preprocess_input,
    shuffle=False
)

# Train with adjusted parameters
epochs = 200

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_rmse",
    patience=30,  # Increased from 10
    restore_best_weights=True,
    verbose=2
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_rmse',
    factor=0.1,  # More aggressive reduction
    patience=3,   # Reduced patience
    min_lr=1e-7,  # Lower minimum learning rate
    verbose=1
)

history = model.fit(train_gen, 
                   validation_data=val_gen,
                   callbacks=[early_stopping, reduce_lr],
                   epochs=epochs, 
                   verbose=2)

# Save the final model (which has the best weights due to restore_best_weights=True)
model.save("live_dmos_resnet50.h5")

# Plot training history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.grid()
plt.legend(fontsize=15)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='valid mae')
plt.grid()
plt.legend(fontsize=15)
plt.show()

# Evaluate model
model.evaluate(test_gen, verbose=2)

# Make predictions
predictions = model.predict(test_gen)
predictions = predictions.flatten()

# Get actual DMOS scores for test set
test_data_df = pd.read_csv(BASE_DIR + 'test_data.csv')
actual_dmos = test_data_df['dmos'].values

print("Sample predictions vs actual:")
for i in range(min(10, len(predictions))):
    print(f"Predicted: {predictions[i]:.3f}, Actual: {actual_dmos[i]:.3f}")

# Calculate correlation
correlation = np.corrcoef(predictions, actual_dmos)[0, 1]
print(f"Correlation coefficient: {correlation:.4f}")

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(actual_dmos, predictions, alpha=0.6)
plt.plot([actual_dmos.min(), actual_dmos.max()], [actual_dmos.min(), actual_dmos.max()], 'r--')
plt.xlabel('Actual DMOS')
plt.ylabel('Predicted DMOS')
plt.title(f'ResNet50 Transfer Learning (Correlation: {correlation:.3f})')
plt.grid(True)
plt.show()

# Show test batch with predictions
show(test_batch, predictions[:4])