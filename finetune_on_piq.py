import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set seed for reproducibility
tf.random.set_seed(1)

# --- Configuration ---
BASE_PIQ_DIR = 'PIQ-Dataset/' 
piq_scores_csv_name = 'Scores_Overall.csv' # Or Scores_Details.csv / Scores_Exposure.csv
piq_image_quality_column = 'JOD' 
PRETRAINED_MODEL_PATH = 'live_dmos_resnet50.h5' # Path to your model trained on LIVE dataset
FINETUNED_MODEL_SAVE_PATH = "live_plus_piq_resnet50_best.h5"

# --- Data Preparation for PIQ Dataset (same as before, with path fix) ---
scores_csv_path = os.path.join(BASE_PIQ_DIR, piq_scores_csv_name)
if not os.path.exists(scores_csv_path):
    print(f"Error: PIQ Scores CSV file not found at {scores_csv_path}")
    exit()

print(f"Loading PIQ scores from: {scores_csv_path}")
piq_data_df = pd.read_csv(scores_csv_path)

required_columns = ['IMAGE PATH', piq_image_quality_column]
for col in required_columns:
    if col not in piq_data_df.columns:
        print(f"Error: Column '{col}' not found in {scores_csv_path}.")
        exit()

def create_filepath(base_directory, img_path_from_csv):
    normalized_csv_path = img_path_from_csv.replace('\\', '/')
    path_components = normalized_csv_path.split('/')
    full_path = os.path.join(base_directory, *path_components)
    return os.path.normpath(full_path)

piq_data_df['filepath'] = piq_data_df['IMAGE PATH'].apply(lambda x: create_filepath(BASE_PIQ_DIR, x))
piq_data_df = piq_data_df[['filepath', piq_image_quality_column]].copy()
piq_data_df.rename(columns={piq_image_quality_column: 'jod_score'}, inplace=True)
piq_data_df.dropna(subset=['filepath', 'jod_score'], inplace=True)

# Debugging filepath existence
print("\n--- Debugging Filepaths for PIQ Dataset ---")
if not piq_data_df.empty:
    print("First 5 constructed filepaths being checked:")
    for p_idx, p_val in enumerate(piq_data_df['filepath'].head().tolist()):
        exists = os.path.exists(p_val)
        print(f"{p_idx}: '{p_val}' -> Exists: {exists}")
else:
    print("DataFrame is empty before path checking for PIQ dataset.")
print("--- End Debugging Filepaths ---\n")

original_count = len(piq_data_df)
piq_data_df = piq_data_df[piq_data_df['filepath'].apply(os.path.exists)]
removed_count = original_count - len(piq_data_df)

if removed_count > 0:
    print(f"Warning: Removed {removed_count} entries from PIQ dataset due to missing image files.")

if piq_data_df.empty:
    print("No valid image data found for PIQ dataset. Please check paths and CSV content.")
    exit()

print(f"Total PIQ images found and with scores: {len(piq_data_df)}")

piq_data_df = piq_data_df.sample(frac=1, random_state=1).reset_index(drop=True)
n_total = len(piq_data_df)
n_train = int(n_total * 0.6)
n_val = int(n_total * 0.25)
n_test = n_total - n_train - n_val

train_df = piq_data_df.iloc[:n_train]
val_df = piq_data_df.iloc[n_train:n_train + n_val]
test_df = piq_data_df.iloc[n_train + n_val:]

print(f"PIQ Dataset for fine-tuning: {n_total} images -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

train_csv_path = os.path.join(BASE_PIQ_DIR, 'piq_finetune_train_data.csv')
val_csv_path = os.path.join(BASE_PIQ_DIR, 'piq_finetune_val_data.csv')
test_csv_path = os.path.join(BASE_PIQ_DIR, 'piq_finetune_test_data.csv')

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"PIQ fine-tune train data CSV saved to: {train_csv_path}")
print(f"PIQ fine-tune validation data CSV saved to: {val_csv_path}")
print(f"PIQ fine-tune test data CSV saved to: {test_csv_path}")
print("="*50)

# --- Custom Data Generator for PIQ Dataset ---
class PIQImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_file, target_size=(224, 224), batch_size=64, 
                 preprocessing_function=None, shuffle=True):
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
        batch_x, batch_y = [], []
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
        if not batch_x: return np.array([]), np.array([])
        batch_x = np.array(batch_x)
        if self.preprocessing_function: batch_x = self.preprocessing_function(batch_x)
        return batch_x, np.array(batch_y, dtype=np.float32)
    
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

# --- Model Loading and Fine-tuning Setup ---
if not os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Error: Pre-trained model '{PRETRAINED_MODEL_PATH}' not found. Please ensure it exists.")
    exit()

print(f"Loading pre-trained model from {PRETRAINED_MODEL_PATH} for fine-tuning...")
model = keras.models.load_model(PRETRAINED_MODEL_PATH)
print("Pre-trained model loaded successfully.")

# --- Configure Layer Trainability for Fine-tuning ---
# Fine-tuning strategy options:
FINETUNE_STRATEGY = "full"  # Options: "full", "top_layers", "last_n_layers"
UNFREEZE_LAST_N_LAYERS = 50  # Only used if strategy is "last_n_layers"

print(f"\nApplying fine-tuning strategy: {FINETUNE_STRATEGY}")

if FINETUNE_STRATEGY == "full":
    # Enable training for all layers
    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, 'layers'):  # For nested models (like ResNet base)
            for sub_layer in layer.layers:
                sub_layer.trainable = True
    print("All layers set to trainable for full fine-tuning.")

elif FINETUNE_STRATEGY == "top_layers":
    # Freeze base model, only train top layers (classifier)
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # This is likely the base ResNet
            layer.trainable = False
        else:  # Top layers (Dense, etc.)
            layer.trainable = True
    print("Base model frozen, only top layers trainable.")

elif FINETUNE_STRATEGY == "last_n_layers":
    # First freeze all layers
    for layer in model.layers:
        layer.trainable = False
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                sub_layer.trainable = False
    
    # Then unfreeze last N layers of the base model
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            base_model = layer
            break
    
    if base_model and len(base_model.layers) >= UNFREEZE_LAST_N_LAYERS:
        for layer in base_model.layers[-UNFREEZE_LAST_N_LAYERS:]:
            layer.trainable = True
        print(f"Last {UNFREEZE_LAST_N_LAYERS} layers of base model set to trainable.")
    
    # Always keep top layers trainable
    for layer in model.layers:
        if not hasattr(layer, 'layers'):  # Top layers
            layer.trainable = True

print("Model summary before re-compilation for fine-tuning:")
model.summary()

# Verify trainability of layers
print("\nTrainable status of layers in loaded model:")
trainable_params = 0
total_params = 0
for layer in model.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")
    if hasattr(layer, 'layers'):  # If it's a model layer itself (like the base ResNet)
        for sub_layer in layer.layers:
            print(f"  Sub-layer: {sub_layer.name}, Trainable: {sub_layer.trainable}")
            if hasattr(sub_layer, 'count_params'):
                params = sub_layer.count_params()
                total_params += params
                if sub_layer.trainable:
                    trainable_params += params
    else:
        if hasattr(layer, 'count_params'):
            params = layer.count_params()
            total_params += params
            if layer.trainable:
                trainable_params += params

print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage trainable: {(trainable_params/total_params)*100:.2f}%")

# --- Compile the loaded model for fine-tuning ---
# Adjust learning rate based on fine-tuning strategy
if FINETUNE_STRATEGY == "full":
    initial_learning_rate = 0.0001  # Lower LR for full fine-tuning
elif FINETUNE_STRATEGY == "top_layers":
    initial_learning_rate = 0.001   # Higher LR for only top layers
else:  # last_n_layers
    initial_learning_rate = 0.0005  # Moderate LR for partial fine-tuning

print(f"Using learning rate: {initial_learning_rate}")

optimizer = keras.optimizers.Adam(
    learning_rate=initial_learning_rate,
    weight_decay=1e-5 
)
loss_fn = keras.losses.MeanSquaredError()
metrics_to_track = ["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')]

model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_to_track)
print("Model re-compiled for fine-tuning on PIQ dataset.")
print("="*50)

# --- Create Data Generators for PIQ fine-tuning ---
IMG_SIZE = (224, 224) # Should match the input size of the loaded model
BATCH_SIZE = 64    # As per original script's training phase
preprocess_input = tf.keras.applications.resnet50.preprocess_input # Assuming ResNet50 base

train_gen = PIQImageDataGenerator(
    csv_file=train_csv_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    preprocessing_function=preprocess_input, shuffle=True
)
val_gen = PIQImageDataGenerator(
    csv_file=val_csv_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    preprocessing_function=preprocess_input, shuffle=False
)
test_gen = PIQImageDataGenerator(
    csv_file=test_csv_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    preprocessing_function=preprocess_input, shuffle=False
)

# --- Callbacks for fine-tuning (mirroring original script's setup) ---
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=FINETUNED_MODEL_SAVE_PATH, save_weights_only=False, 
    monitor='val_rmse', mode='min', save_best_only=True, verbose=1
)
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_rmse", patience=30, restore_best_weights=True, verbose=2
)
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_rmse', factor=0.1, patience=3, min_lr=1e-7, verbose=1
)

# --- Fine-tune the model on PIQ dataset ---
epochs = 200 # Epochs from original script
print(f"Starting fine-tuning on PIQ dataset for {epochs} epochs...")
history = model.fit(
    train_gen, validation_data=val_gen, epochs=epochs,
    callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback],
    verbose=2
)

print(f"Fine-tuning complete. Best model saved to {FINETUNED_MODEL_SAVE_PATH}")
model.load_weights(FINETUNED_MODEL_SAVE_PATH) # Load the best weights

# --- Plot training history ---
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.grid(); plt.legend(fontsize=15); plt.title('Loss (Mean Squared Error)')
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
if 'rmse' in history.history: # RMSE might not be in history if not compiled with it (but it is here)
    plt.plot(history.history['rmse'], label='Train RMSE')
    plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.grid(); plt.legend(fontsize=15); plt.title('MAE & RMSE')
plt.show()

# --- Evaluate fine-tuned model ---
print("Evaluating fine-tuned model on the PIQ test set:")
model.evaluate(test_gen, verbose=2)

# --- Make predictions ---
print("Making predictions on the PIQ test set...")
predictions = model.predict(test_gen).flatten()
actual_jod_scores = pd.read_csv(test_csv_path)['jod_score'].values
min_len = min(len(predictions), len(actual_jod_scores))
predictions = predictions[:min_len]; actual_jod_scores = actual_jod_scores[:min_len]

print("\nSample predictions vs actual JOD scores (fine-tuned model):")
for i in range(min(10, len(predictions))):
    print(f"Predicted: {predictions[i]:.3f}, Actual: {actual_jod_scores[i]:.3f}")

if len(predictions) > 1 and len(actual_jod_scores) > 1:
    correlation = np.corrcoef(predictions, actual_jod_scores)[0, 1]
    print(f"\nPearson Correlation (fine-tuned model): {correlation:.4f}")
    plt.figure(figsize=(8, 8))
    sn.scatterplot(x=actual_jod_scores, y=predictions, alpha=0.6)
    plt.plot([min(actual_jod_scores.min(), predictions.min()), max(actual_jod_scores.max(), predictions.max())],
             [min(actual_jod_scores.min(), predictions.min()), max(actual_jod_scores.max(), predictions.max())],
             color='red', linestyle='--')
    plt.xlabel("Actual JOD Scores"); plt.ylabel("Predicted JOD Scores")
    plt.title(f"Fine-tuned Model: Predictions vs Actual JOD (Corr: {correlation:.4f})"); plt.grid(True); plt.show()
else:
    print("\nNot enough data points for correlation or scatter plot.")

print("\nFine-tuning process complete.")