# Rice Leaf Disease Detector — Trained From Scratch (No Pretrained Model)

This project does **not** use any pretrained/downloaded Keras model (no VGG16, no MobileNetV2, no ImageNet weights). The CNN is defined and trained entirely from scratch on your own dataset using `train_model.py`, then served through a Flask web app.

## 📁 Project Structure

```
rice_leaf_scratch_app/
├── train_model.py          ← builds & trains the CNN from scratch, saves the model
├── app.py                  ← Flask web app that serves the trained model
├── requirements.txt
├── dataset/                 ← put your raw images here (see below)
│   ├── Bacterial Leaf Blight/
│   ├── Brown Spot/
│   └── Leaf Smut/
├── templates/
│   └── index.html           ← green/white/brown themed UI
└── static/
    └── uploads/              ← uploaded prediction images are saved here
```

After training, two new files/folders are created automatically:
- `Rice_Leaf_Disease_Scratch_Model.keras` — the trained model
- `class_names.json` — the correct class label order (read automatically by `app.py`)
- `dataset_split/` — the train/val/test split created from your `dataset/` folder

## ⚙️ Setup

1. **Open this folder in VS Code.**

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   ```
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your dataset.**
   Place your images inside `dataset/`, organized as one folder per class:
   ```
   dataset/
   ├── Bacterial Leaf Blight/
   │   ├── img1.jpg
   │   └── ...
   ├── Brown Spot/
   │   └── ...
   └── Leaf Smut/
       └── ...
   ```

## 🏋️ Step 1 — Train the Model From Scratch

```bash
python train_model.py
```

This will:
- Split your dataset into train/val/test (70/15/15)
- Apply real-time data augmentation (rotation, zoom, shift, flip)
- Build a CNN from scratch (4 convolution blocks + dense layers, no pretrained weights)
- Train with EarlyStopping + learning-rate reduction
- Evaluate on the held-out test set and print the final accuracy
- Save `Rice_Leaf_Disease_Scratch_Model.keras` and `class_names.json`

Training time depends on your machine and dataset size — this can take anywhere from a few minutes to longer on CPU-only machines.

## ▶️ Step 2 — Run the Web App

Once training finishes:

```bash
python app.py
```

Open your browser at:

```
http://127.0.0.1:3000
```

Upload a rice leaf photo and click **Predict Disease**.

## 🛑 Stop the Server

Press `Ctrl + C` in the terminal.

## Notes

- Because the model is trained entirely from scratch (no transfer learning), it needs a reasonably sized dataset to generalize well — a very small dataset (a few dozen images per class) may still overfit even with augmentation. If accuracy is low, the most effective fix is collecting more training images per class.
- `app.py` will refuse to start with a clear error message if `Rice_Leaf_Disease_Scratch_Model.keras` doesn't exist yet — always run `train_model.py` first.
- To change the port, edit the `port=3000` line at the bottom of `app.py`.
