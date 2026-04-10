"""
train_model.py — Optimized Production training script for Pneumonia Detection (Fast Profile).

Usage:
    python train_model.py

Expects: Kaggle chest_xray dataset in ./chest_xray/
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

ROOT = os.path.dirname(os.path.abspath(__file__))


def _save_plot(history, key: str, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    if key in history.history:
        plt.plot(history.history[key], label=f"Train {key}", linewidth=2)
    val_key = f"val_{key}"
    if val_key in history.history:
        plt.plot(history.history[val_key], label=f"Val {key}", linewidth=2)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(key.capitalize())
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(ROOT, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def _gradcam_sample(model: tf.keras.Model, test_ds: tf.data.Dataset) -> None:
    # Attempt to find the last conv layer for Grad-CAM
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv_name = layer.name
            break
    
    if not last_conv_name:
        # Check inside backbone
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                for sub in reversed(layer.layers):
                    if "Conv" in sub.__class__.__name__:
                        last_conv_name = sub.name
                        break
    
    if not last_conv_name:
        return

    for images, labels in test_ds.take(1):
        img_tensor = images[:1]
        break
    else: return

    try:
        # Create a model that outputs the activation and the prediction
        # For MobileNetV2, we target the backbone's layers if necessary
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(index=4).output, model.output] # Index 4 is usually the backbone
        )
        
        with tf.GradientTape() as tape:
            activations, preds = grad_model(img_tensor)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]
        
        grads = tape.gradient(loss, activations)
        if grads is None: return
        
        weights = tf.reduce_mean(grads[0], axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, activations[0]), axis=-1)
        cam = np.maximum(cam.numpy(), 0)
        if cam.max() > 0: cam = cam / cam.max()
        
        from PIL import Image
        base_img = Image.fromarray((img_tensor[0].numpy() * 127.5 + 127.5).astype(np.uint8)).resize((160, 160))
        cam_heat = np.array(Image.fromarray(np.uint8(cam * 255), mode="L").resize((160, 160)), dtype=np.float32)
        
        heatmap = plt.cm.jet(cam_heat / 255.0)[:, :, :3] * 255
        overlay = (0.6 * np.array(base_img) + 0.4 * heatmap).clip(0, 255).astype(np.uint8)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.imshow(base_img); plt.title("Input"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(overlay); plt.title("Grad-CAM overlay"); plt.axis("off")
        plt.savefig(os.path.join(ROOT, "gradcam_sample.png"), dpi=150)
        plt.close()
        print("  Saved gradcam_sample.png")
    except Exception:
        pass


def main() -> None:
    data_root = os.path.join(ROOT, "chest_xray")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    img_size = (160, 160)
    batch_size = 64
    seed = 1337

    if not os.path.isdir(train_dir):
        raise SystemExit("Dataset folder not found at ./chest_xray/train")

    # ── Data loading ──────────────────────────────────────────────
    params = dict(image_size=img_size, batch_size=batch_size, seed=seed, label_mode="categorical")
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, **params)
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, **params)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, **params)

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # ── Class weights ─────────────────────────────────────────────
    print("Computing class weights...")
    y_true = []
    for _, labels in train_ds.take(20):
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
    
    weights = compute_class_weight("balanced", classes=np.unique(y_true), y=np.array(y_true))
    class_weights = dict(enumerate(weights))
    print(f"  Weights: {class_weights}")

    # ── Augmentation & Prefetch ───────────────────────────────────
    aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.05)])
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1024).map(lambda x, y: (aug(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds_pf = test_ds.prefetch(AUTOTUNE)

    # ── Model ─────────────────────────────────────────────────────
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=img_size + (3,))
    base.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # ── Phase 1: Head Only ────────────────────────────────────────
    print("\n═══ Phase 1: Head Training (3 Epochs) ═══")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy", "auc"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=3, class_weight=class_weights)

    # ── Phase 2: Fine-tuning ──────────────────────────────────────
    print("\n═══ Phase 2: Fine-tuning (7 Epochs) ═══")
    base.trainable = True
    for layer in base.layers[:-20]: layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy", "auc"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=7, class_weight=class_weights)

    # ── Evaluation & Curves ───────────────────────────────────────
    merged = {k: h1.history[k] + h2.history[k] for k in h1.history}
    class Hist:
        def __init__(self, d): self.history = d
    _save_plot(Hist(merged), "accuracy", "Accuracy", "accuracy_curve.png")
    _save_plot(Hist(merged), "auc", "AUC", "auc_curve.png")

    print("\n═══ Final Evaluation ═══")
    model.evaluate(test_ds_pf, verbose=2)

    y_test, y_pred_probs = [], []
    for imgs, lbls in test_ds_pf:
        y_test.extend(lbls.numpy().tolist())
        y_pred_probs.extend(model.predict(imgs, verbose=0).tolist())
    
    y_test, y_pred_probs = np.array(y_test), np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true_idx = np.argmax(y_test, axis=1)

    rep = classification_report(y_true_idx, y_pred, target_names=class_names, output_dict=True)
    print("\n" + classification_report(y_true_idx, y_pred, target_names=class_names))
    auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr")
    print(f"Final AUC-ROC: {auc:.4f}")

    # ── Save Results ──────────────────────────────────────────────
    with open(os.path.join(ROOT, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": round(rep["accuracy"] * 100, 1),
            "precision": round(rep["weighted avg"]["precision"] * 100, 1),
            "recall": round(rep["weighted avg"]["recall"] * 100, 1),
            "f1_score": round(rep["weighted avg"]["f1-score"] * 100, 1),
            "auc_roc": round(auc, 4),
            "backbone": "MobileNetV2 (Fast)",
            "input_size": "160x160"
        }, f, indent=2)

    model.save(os.path.join(ROOT, "model.keras"))
    print(f"\n✅ Training Complete. Model saved to model.keras")
    
    _gradcam_sample(model, test_ds_pf)


if __name__ == "__main__":
    np.random.seed(1337)
    tf.random.set_seed(1337)
    main()
