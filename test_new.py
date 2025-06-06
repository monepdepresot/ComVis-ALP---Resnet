import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
# Replace this path by wherever you saved your .h5 file
MODEL_PATH = "live_plus_piq_resnet50_best.h5"
model = load_model(MODEL_PATH)

TARGET_SIZE = (224, 224)

def load_and_preprocess(img_path):
    """
    1) Loads the image from disk
    2) Resizes to TARGET_SIZE
    3) Converts to array
    4) Applies ResNet50 preprocess_input (scales to [-1, +1], etc.)
    """
    img = load_img(img_path, target_size=TARGET_SIZE)
    x   = img_to_array(img)                 # shape = (224, 224, 3)
    x   = np.expand_dims(x, axis=0)          # shape = (1, 224, 224, 3)
    x   = preprocess_input(x)                # ResNet50‐style scaling
    return x
IPPA_DIR    = "images_by_grade"          # top‐level directory containing “a/”, “b/”, “c/”, “d/”
SUBFOLDERS  = ["a", "b", "c", "d"]
IMAGE_EXTS  = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

rows = []

for sub in SUBFOLDERS:
    subdir = os.path.join(IPPA_DIR, sub)
    if not os.path.isdir(subdir):
        print(f"Warning: {subdir} does not exist or is not a folder.")
        continue

    for fname in os.listdir(subdir):
        lower = fname.lower()
        if not lower.endswith(IMAGE_EXTS):
            continue

        img_path = os.path.join(subdir, fname)
        try:
            x_input = load_and_preprocess(img_path)   # shape = (1,224,224,3)
            pred    = model.predict(x_input, verbose=0)  # shape = (1, 1)
            dmos    = float(pred[0, 0])               # get scalar DMOS
        except Exception as e:
            print(f"Could not process {img_path}: {e}")
            continue

        rows.append({
            "subfolder": sub,
            "filename": fname,
            "predicted_dmos": dmos
        })

# Convert to DataFrame so that it’s easy to inspect / group / visualize
df = pd.DataFrame(rows)
print(df.head())

if MODEL_PATH != "live_dmos_resnet50.h5":
    print("Using MOS model, not DMOS model.")
    summary = df.groupby("subfolder")["predicted_mos"].agg(
        ["count", "mean", "std", "min", "max"]
    )
else:
    summary = df.groupby("subfolder")["predicted_dmos"].agg(
        ["count", "mean", "std", "min", "max"]
    )


print("Model: ", MODEL_PATH)
print(summary)

