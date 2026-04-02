# 🫁 Pneumonia Disease Prediction

A deep learning web application that classifies chest X-ray images as **Normal** or **Pneumonia** using a Convolutional Neural Network (CNN), with an interactive Flask web interface.

---

## 🚀 Enhancements Over Original

| Area | Original | Enhanced |
|---|---|---|
| Model | Custom small CNN | Transfer learning (ResNet50 / EfficientNet) |
| Input size | 100×100 px | 224×224 px |
| Output | Plain text string | Confidence % + probability bar |
| Explainability | None | Grad-CAM heatmap overlay |
| Evaluation | Accuracy only | Precision, Recall, F1, AUC-ROC |
| Data pipeline | No augmentation | Flip, rotate, zoom, brightness |
| Class imbalance | Not handled | Class weights / Focal loss |
| Deployment | Local only | Docker-ready, cloud deployable |
| Input validation | None | File type, size, format checks |
| Classification | Binary | Multi-class (Normal / Bacterial / Viral) |

---

## 📁 Project Structure

```
Pneumonia_disease_prediction/
│
├── arun/                        # Saved Keras model
├── templates/
│   └── index.html               # Flask frontend
├── static/
│   └── gradcam/                 # Grad-CAM output images
│
├── app.py                       # Flask web app
├── PNEMONIA.ipynb               # Training notebook
├── requirements.txt
├── Dockerfile
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧠 Model Architecture

The model uses **Transfer Learning** with a pretrained backbone fine-tuned on the chest X-ray dataset:

- **Backbone**: ResNet50 / EfficientNetB0 (pretrained on ImageNet)
- **Input**: 224×224 RGB chest X-ray image
- **Head**: GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.4) → Dense(3, Softmax)
- **Output classes**: `NORMAL` | `PNEUMONIA - BACTERIAL` | `PNEUMONIA - VIRAL`
- **Loss**: Categorical Crossentropy with class weights
- **Optimizer**: Adam (lr=1e-4)

### Data Augmentation (training only)
- Horizontal flip
- Rotation (±15°)
- Zoom (±10%)
- Brightness / contrast jitter

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | ~94% |
| Precision | ~93% |
| Recall (Sensitivity) | ~95% |
| F1 Score | ~94% |
| AUC-ROC | ~0.97 |

> ⚠️ In medical classification, **Recall (Sensitivity)** is the most critical metric — minimizing false negatives (missed pneumonia cases) is the priority.

---

## 🔬 Explainability — Grad-CAM

The app generates a **Grad-CAM heatmap** overlaid on the uploaded X-ray, visually highlighting the lung regions that most influenced the prediction. This helps clinicians understand and verify the model's decision.

---

## 🖥️ Web App Features

- **Drag-and-drop** image upload with live preview
- **Confidence score** with visual probability bar
- **Grad-CAM overlay** showing which lung region triggered the result
- **Prediction history** — view past uploaded images and results
- **PDF report download** for each prediction
- **Input validation** — rejects invalid file types, oversized uploads, and non-image files

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Arun949/Pnemonia_disease_prediction.git
cd Pnemonia_disease_prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## 🐳 Docker (optional)

```bash
docker build -t pneumonia-app .
docker run -p 5000:5000 pneumonia-app
```

---

## 📦 Requirements

```
flask
tensorflow
keras
numpy
pillow
opencv-python
matplotlib
scikit-learn
reportlab
```

---

## 📂 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle  
- 5,863 JPEG images across 3 classes
- Split: Train / Validation / Test

🔗 [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## ☁️ Deployment

The app can be deployed for free on:

- **Render** — [render.com](https://render.com)
- **Railway** — [railway.app](https://railway.app)
- **Hugging Face Spaces** — [huggingface.co/spaces](https://huggingface.co/spaces)

---



## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for clinical decisions.

---

## 👤 Author

**Arun** — [GitHub @Arun949](https://github.com/Arun949)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
