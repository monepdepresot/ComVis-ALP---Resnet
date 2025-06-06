import os
import math
import random
# import shutil # Not used for copying images in this version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
# from scipy.io import loadmat # Not needed for PIQ dataset in CSV format

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set seed for reproducibility
tf.random.set_seed(1)

# --- PIQ Dataset Configuration ---
BASE_DIR = 'PIQ-Dataset/' 
piq_scores_csv_name = 'Scores_Overall.csv' # You can change this to Scores_Details.csv or Scores_Exposure.csv
piq_image_quality_column = 'JOD' 

# --- Data Preparation for PIQ Dataset ---
scores_csv_path = os.path.join(BASE_DIR, piq_scores_csv_name)
if not os.path.exists(scores_csv_path):
    print(f"Error: Scores CSV file not found at {scores_csv_path}")
    exit()

print(f"Loading scores from: {scores_csv_path}")
piq_data_df = pd.read_csv(scores_csv_path)

# Debug: Show first few IMAGE PATH entries
print(f"Sample IMAGE PATH entries from CSV:")
print(piq_data_df['IMAGE PATH'].head(10).tolist())

required_columns = ['IMAGE PATH', piq_image_quality_column]
for col in required_columns:
    if col not in piq_data_df.columns:
        print(f"Error: Column '{col}' not found in {scores_csv_path}.")
        print(f"Available columns: {piq_data_df.columns.tolist()}")
        exit()

# Debug: Check if BASE_DIR exists and what's inside
print(f"BASE_DIR exists: {os.path.exists(BASE_DIR)}")
if os.path.exists(BASE_DIR):
    print(f"Contents of BASE_DIR: {os.listdir(BASE_DIR)[:10]}")  # Show first 10 items

# Fix path separators for cross-platform compatibility
def normalize_path(image_path):
    """Convert Windows-style paths to Unix-style paths"""
    # Replace backslashes with forward slashes
    normalized = image_path.replace('\\', '/')
    return normalized

piq_data_df['IMAGE PATH'] = piq_data_df['IMAGE PATH'].apply(normalize_path)
piq_data_df['filepath'] = piq_data_df['IMAGE PATH'].apply(lambda x: os.path.join(BASE_DIR, x))

# Debug: Show sample generated filepaths
print(f"Sample generated filepaths:")
sample_paths = piq_data_df['filepath'].head(5).tolist()
for path in sample_paths:
    print(f"  {path} -> exists: {os.path.exists(path)}")

# Check for common image extensions if original path doesn't exist
def find_image_with_extensions(base_path):
    """Try to find image with common extensions if exact path doesn't exist"""
    if os.path.exists(base_path):
        return base_path
    
    # Remove extension and try common image extensions
    base_without_ext = os.path.splitext(base_path)[0]
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    for ext in extensions:
        candidate_path = base_without_ext + ext
        if os.path.exists(candidate_path):
            return candidate_path
    
    return None

# Apply the robust path finding
piq_data_df['filepath_original'] = piq_data_df['filepath']
piq_data_df['filepath'] = piq_data_df['filepath'].apply(find_image_with_extensions)

piq_data_df = piq_data_df[['filepath', piq_image_quality_column]].copy()
piq_data_df.rename(columns={piq_image_quality_column: 'jod_score'}, inplace=True)
piq_data_df.dropna(subset=['filepath', 'jod_score'], inplace=True)

# Remove entries where filepath is None (couldn't find image)
original_count = len(piq_data_df)
piq_data_df = piq_data_df[piq_data_df['filepath'].notna()]
piq_data_df = piq_data_df[piq_data_df['filepath'].apply(lambda x: x is not None and os.path.exists(x))]

if len(piq_data_df) < original_count:
    print(f"Warning: Removed {original_count - len(piq_data_df)} entries due to missing image files.")
    
    # Show some examples of missing files for debugging
    missing_files = []
    for i, row in pd.read_csv(scores_csv_path).head(10).iterrows():
        img_path = os.path.join(BASE_DIR, row['IMAGE PATH'])
        if not os.path.exists(img_path):
            missing_files.append(img_path)
    
    if missing_files:
        print(f"Examples of missing files:")
        for missing in missing_files[:5]:
            print(f"  {missing}")

if piq_data_df.empty:
    print("No valid image data found. Please check paths and CSV content.")
    print(f"Suggestions:")
    print(f"1. Verify that images are in the correct subdirectory within {BASE_DIR}")
    print(f"2. Check if IMAGE PATH in CSV contains the correct relative paths")
    print(f"3. Ensure image files have the expected extensions")
    exit()

print(f"Total images found and with scores: {len(piq_data_df)}")

piq_data_df = piq_data_df.sample(frac=1, random_state=1).reset_index(drop=True)

n_total = len(piq_data_df)
n_train = int(n_total * 0.6)
n_val = int(n_total * 0.25) # Original script implies 0.2 for validation from 0.6 train, 0.2 test. Adjusted for a common 60/25/15 split
n_test = n_total - n_train - n_val

train_df = piq_data_df.iloc[:n_train]
val_df = piq_data_df.iloc[n_train:n_train + n_val]
test_df = piq_data_df.iloc[n_train + n_val:]

print(f"PIQ Dataset: {n_total} images -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

train_csv_path = os.path.join(BASE_DIR, 'piq_train_data.csv')
val_csv_path = os.path.join(BASE_DIR, 'piq_val_data.csv')
test_csv_path = os.path.join(BASE_DIR, 'piq_test_data.csv')

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Train data CSV saved to: {train_csv_path}")
print(f"Validation data CSV saved to: {val_csv_path}")
print(f"Test data CSV saved to: {test_csv_path}")
print("="*50)

# --- Custom Data Generator for PIQ Dataset (adapted from DMOSImageDataGenerator) ---
class PIQImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_file, target_size=(224, 224), batch_size=64, 
                 preprocessing_function=None, shuffle=True): # Default batch_size to 64
        self.data = pd.read_csv(csv_file)
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
        batch_data = self.data.iloc[batch_indices]
        
        batch_x = []
        batch_y = []

        for i, row in batch_data.iterrows():
            img_path = row['filepath']
            try:
                img = keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
                img_array = keras.preprocessing.image.img_to_array(img)
                batch_x.append(img_array)
                batch_y.append(row['jod_score']) 
            except Exception as e:
                print(f"Warning: Skipping image {img_path} due to error: {e}")
                continue
        
        if not batch_x:
             print(f"Warning: Batch {idx} is empty after trying to load images.")
             return np.array([]), np.array([])

        batch_x = np.array(batch_x)
        if self.preprocessing_function:
            batch_x = self.preprocessing_function(batch_x)
        # else: # Original script did not have this else for the training generators
        #     img_array = img_array / 255.0 
                
        return batch_x, np.array(batch_y, dtype=np.float32)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# --- Model Definition (Mirroring original script's ResNet50 setup) ---
def get_piq_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'  # Average pooling at the end of ResNet50
    )
    # Freeze all layers of the base model
    for layer in base_model.layers:
        layer.trainable = True

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False) 
    # Directly connect to the output layer for regression
    outputs = layers.Dense(1, activation='linear')(x) # Linear activation for regression
    model = keras.Model(inputs, outputs)
    return model

# Get preprocessing function for ResNet50
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# --- Create Data Generators ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 64 # As per original script's training phase

train_gen = PIQImageDataGenerator(
    csv_file=train_csv_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    preprocessing_function=preprocess_input, # Apply ResNet50 preprocessing
    shuffle=True
)

val_gen = PIQImageDataGenerator(
    csv_file=val_csv_path, 
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    preprocessing_function=preprocess_input, # Apply ResNet50 preprocessing
    shuffle=False
)

test_gen = PIQImageDataGenerator(
    csv_file=test_csv_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, # Consistent batch size for testing
    preprocessing_function=preprocess_input, # Apply ResNet50 preprocessing
    shuffle=False
)

# --- Model Training ---
model = get_piq_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Compile the model (mirroring original script's settings)
loss_fn = keras.losses.MeanSquaredError()
initial_learning_rate = 0.001
optimizer = keras.optimizers.Adam(
    learning_rate=initial_learning_rate,
    weight_decay=1e-5 
)
metrics_to_track = ["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')]

model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_to_track)

model.summary()

# Callbacks (mirroring original script's setup)
checkpoint_filepath = "piq_jod_resnet50_best.h5" # Updated model name
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False, 
    monitor='val_rmse', # Monitor validation RMSE as in original's EarlyStopping
    mode='min',
    save_best_only=True,
    verbose=1 # Added verbose for checkpoint saving
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_rmse", # Monitor validation RMSE
    patience=30,        # Patience from original script
    restore_best_weights=True,
    verbose=2
)

reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_rmse', # Monitor validation RMSE
    factor=0.1,         # Factor from original script
    patience=3,         # Patience from original script
    min_lr=1e-7,        # Min LR from original script
    verbose=1
)

# Train the model
epochs = 200 # Epochs from original script
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback],
    verbose=2
)


# Load best model for evaluation (though restore_best_weights=True should handle this for EarlyStopping)
print("Loading best model weights from:", checkpoint_filepath)
model.load_weights(checkpoint_filepath)

# Plot training history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.grid()
plt.legend(fontsize=15)
plt.title('Loss (Mean Squared Error)')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.plot(history.history['rmse'], label='Train RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.grid()
plt.legend(fontsize=15)
plt.title('Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)')
plt.show()

# Evaluate model
print("Evaluating model on the test set:")
model.evaluate(test_gen, verbose=2)

# Make predictions
print("Making predictions on the test set...")
predictions = model.predict(test_gen)
predictions = predictions.flatten()

# Get actual JOD scores for test set
actual_jod_scores = pd.read_csv(test_csv_path)['jod_score'].values

min_len = min(len(predictions), len(actual_jod_scores))
predictions = predictions[:min_len]
actual_jod_scores = actual_jod_scores[:min_len]

print("\nSample predictions vs actual JOD scores:")
for i in range(min(10, len(predictions))):
    print(f"Predicted: {predictions[i]:.3f}, Actual: {actual_jod_scores[i]:.3f}")

# Calculate correlation
if len(predictions) > 1 and len(actual_jod_scores) > 1:
    correlation = np.corrcoef(predictions, actual_jod_scores)[0, 1]
    print(f"\nPearson Correlation between predictions and actual JOD scores: {correlation:.4f}")

    # Scatter plot of predictions vs actual
    plt.figure(figsize=(8, 8))
    sn.scatterplot(x=actual_jod_scores, y=predictions, alpha=0.6)
    plt.plot([min(actual_jod_scores.min(), predictions.min()), max(actual_jod_scores.max(), predictions.max())], 
             [min(actual_jod_scores.min(), predictions.min()), max(actual_jod_scores.max(), predictions.max())], 
             color='red', linestyle='--') # Ideal line
    plt.xlabel("Actual JOD Scores")
    plt.ylabel("Predicted JOD Scores")
    plt.title(f"Predictions vs Actual JOD (Correlation: {correlation:.4f})")
    plt.grid(True)
    plt.show()
else:
    print("\nNot enough data points to calculate correlation or plot scatter.")

print("\nTraining and evaluation complete.")