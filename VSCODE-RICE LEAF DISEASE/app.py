from flask import Flask, render_template, request
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
MODEL_PATH = "Rice_Leaf_Disease_Scratch_Model.keras"   # produced by train_model.py
CLASS_NAMES_FILE = "class_names.json"                   # produced by train_model.py
UPLOAD_FOLDER = os.path.join("static", "uploads")
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------------------------------
# Check the trained model exists before starting
# ---------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n\n'{MODEL_PATH}' not found.\n"
        "This app does NOT ship with a pretrained/downloaded model - "
        "you must train it yourself first by running:\n\n"
        "    python train_model.py\n\n"
        "Make sure your dataset is organized inside the 'dataset/' folder "
        "(one sub-folder per class) before running that script.\n"
    )

if not os.path.exists(CLASS_NAMES_FILE):
    raise FileNotFoundError(
        f"'{CLASS_NAMES_FILE}' not found. Run 'python train_model.py' first "
        "so the class label order is saved correctly."
    )

# ---------------------------------------------------------------
# Load the model trained from scratch by train_model.py
# ---------------------------------------------------------------
print("Loading model trained from scratch:", MODEL_PATH)
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_FILE) as f:
    CLASS_NAMES = json.load(f)

print("Class names (in model output order):", CLASS_NAMES)


def predict_image(img_path):
    """Loads an image from disk, preprocesses it, and returns
    (predicted_class_name, confidence_percentage)."""
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)
    return predicted_class, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("leaf_image")
        if file and file.filename:
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            result, confidence = predict_image(save_path)
            image_path = os.path.join("uploads", filename).replace("\\", "/")

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path,
    )


if __name__ == "__main__":
    # Runs on http://127.0.0.1:3000
    app.run(host="0.0.0.0", port=3000, debug=True)
