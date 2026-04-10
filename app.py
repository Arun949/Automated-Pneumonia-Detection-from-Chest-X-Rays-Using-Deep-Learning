from __future__ import annotations

import base64
import io
import json
import os
import secrets
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, make_response, render_template, request, session
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "arun")
FALLBACK_MODEL_PATH = os.path.join(APP_DIR, "fallback_model.keras")
KERAS_MODEL_PATH = os.path.join(APP_DIR, "model.keras")
METRICS_PATH = os.path.join(APP_DIR, "metrics.json")

IMG_SIZE: Tuple[int, int] = (160, 160)
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5MB

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES

# Keep recent prediction payloads in memory for PDF report downloads.
# Avoid storing base64 images in client sessions.
REPORT_CACHE_MAX = 25
REPORT_CACHE_TTL_S = 10 * 60
_REPORT_CACHE: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()


def _report_cache_put(report_id: str, payload: Dict[str, Any]) -> None:
    now = time.time()
    _REPORT_CACHE[report_id] = (now, payload)
    _REPORT_CACHE.move_to_end(report_id, last=True)
    # Evict old by size
    while len(_REPORT_CACHE) > REPORT_CACHE_MAX:
        _REPORT_CACHE.popitem(last=False)
    # Evict old by TTL
    cutoff = now - REPORT_CACHE_TTL_S
    stale = [k for k, (ts, _) in _REPORT_CACHE.items() if ts < cutoff]
    for k in stale:
        _REPORT_CACHE.pop(k, None)


def _report_cache_get(report_id: str) -> Optional[Dict[str, Any]]:
    item = _REPORT_CACHE.get(report_id)
    if item is None:
        return None
    ts, payload = item
    if time.time() - ts > REPORT_CACHE_TTL_S:
        _REPORT_CACHE.pop(report_id, None)
        return None
    _REPORT_CACHE.move_to_end(report_id, last=True)
    return payload


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    # The model's internal Rescaling layer expects raw pixels [0, 255].
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ── Load metrics.json if available ────────────────────────────────
def _load_metrics() -> Optional[Dict[str, Any]]:
    if os.path.isfile(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return None


SAVED_METRICS = _load_metrics()


def _load_model() -> tuple[tf.keras.Model, str]:
    # Prefer native Keras formats; fall back to SavedModel via TFSMLayer (Keras 3).
    if os.path.isfile(KERAS_MODEL_PATH):
        return tf.keras.models.load_model(KERAS_MODEL_PATH), "model.keras"

    if os.path.isfile(MODEL_PATH) and MODEL_PATH.endswith((".keras", ".h5")):
        return tf.keras.models.load_model(MODEL_PATH), os.path.basename(MODEL_PATH)

    # If it's a directory, assume SavedModel (common in older projects).
    if os.path.isdir(MODEL_PATH):
        try:
            layer = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
            inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
            outputs = layer(inputs)
            if isinstance(outputs, dict):
                # SavedModels often return a dict like {"output_0": ...}
                outputs = next(iter(outputs.values()))
            return tf.keras.Model(inputs, outputs), "arun/ (SavedModel)"
        except Exception:
            pass

    # If the project model is missing/corrupted, fall back to a known-good Keras file.
    if os.path.isfile(FALLBACK_MODEL_PATH):
        return tf.keras.models.load_model(FALLBACK_MODEL_PATH), "fallback_model.keras (demo)"

    # Final fallback: let Keras attempt to load.
    return tf.keras.models.load_model(MODEL_PATH), os.path.basename(MODEL_PATH)


MODEL, MODEL_SOURCE = _load_model()


def _infer_class_names(model: tf.keras.Model) -> List[str]:
    try:
        out_shape = model.output_shape  # type: ignore[attr-defined]
        n = int(out_shape[-1]) if isinstance(out_shape, (list, tuple)) else 2
    except Exception:
        n = 2

    if n == 3:
        return ["NORMAL", "PNEUMONIA – BACTERIAL", "PNEUMONIA – VIRAL"]
    if n == 2:
        return ["NORMAL", "PNEUMONIA"]
    return [f"CLASS_{i}" for i in range(n)]


CLASS_NAMES = _infer_class_names(MODEL)


def _find_last_conv_layer_name(model: tf.keras.Model) -> Optional[str]:
    # TFSMLayer-wrapped SavedModels don't expose inner layers for Grad-CAM.
    if any(isinstance(l, tf.keras.layers.TFSMLayer) for l in model.layers):
        return None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


LAST_CONV_LAYER = _find_last_conv_layer_name(MODEL)


def _jet_colormap(intensity: float) -> Tuple[int, int, int]:
    """Simple JET-like colourmap: blue → cyan → green → yellow → red."""
    v = max(0.0, min(1.0, intensity))
    r = min(1.0, max(0.0, 1.5 - abs(v - 0.75) * 4))
    g = min(1.0, max(0.0, 1.5 - abs(v - 0.5) * 4))
    b = min(1.0, max(0.0, 1.5 - abs(v - 0.25) * 4))
    return int(r * 255), int(g * 255), int(b * 255)


def _apply_jet_colormap(cam_np: np.ndarray) -> np.ndarray:
    """Apply a JET colourmap to a single-channel (H, W) float array in [0, 1]."""
    h, w = cam_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            rgb[i, j] = _jet_colormap(cam_np[i, j])
    return rgb


def _make_gradcam_overlay(img: Image.Image, img_array: np.ndarray) -> Optional[Image.Image]:
    if LAST_CONV_LAYER is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs=MODEL.inputs,
            outputs=[MODEL.get_layer(LAST_CONV_LAYER).output, MODEL.output],
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        conv_out0 = conv_out[0]
        grads0 = grads[0]
        weights = tf.reduce_mean(grads0, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_out0), axis=-1)
        cam = tf.maximum(cam, 0)
        cam_np = cam.numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        # Resize CAM to image size
        cam_img = Image.fromarray(np.uint8(cam_np * 255.0), mode="L").resize(img.size)
        cam_float = np.array(cam_img, dtype=np.float32) / 255.0

        # Apply JET colourmap (no OpenCV dependency)
        heat_rgb = _apply_jet_colormap(cam_float)

        base = img.convert("RGB")
        base_np = np.array(base, dtype=np.float32)
        overlay = (0.6 * base_np + 0.4 * heat_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception:
        return None


def _now_ts() -> str:
    return time.strftime("%H:%M  %d %b %Y")


# ── Build model info dict for the template ────────────────────────
def _build_model_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "input_size": f"{IMG_SIZE[0]}×{IMG_SIZE[1]}",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "source": MODEL_SOURCE,
        "is_demo": "demo" in MODEL_SOURCE.lower() or "fallback" in MODEL_SOURCE.lower(),
    }
    if SAVED_METRICS:
        info["accuracy"] = SAVED_METRICS.get("accuracy")
        info["precision"] = SAVED_METRICS.get("precision")
        info["recall"] = SAVED_METRICS.get("recall")
        info["f1_score"] = SAVED_METRICS.get("f1_score")
        info["auc_roc"] = SAVED_METRICS.get("auc_roc")
        info["backbone"] = SAVED_METRICS.get("backbone")
    return info


MODEL_INFO = _build_model_info()


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_info=MODEL_INFO,
        history=session.get("history", []),
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_source": MODEL_SOURCE,
        "classes": CLASS_NAMES,
    })


@app.post("/clear-history")
def clear_history():
    session["history"] = []
    return jsonify({"ok": True})


@app.get("/report/<report_id>")
def report(report_id: str):
    payload = _report_cache_get(report_id)
    if payload is None:
        return jsonify({"error": "Report not found or expired. Run a new prediction."}), 404

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except Exception:
        return jsonify({"error": "Missing dependency: reportlab. Install requirements.txt again."}), 500

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    c.setTitle(f"Pneumonia Detection Report {report_id}")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(48, h - 56, "Pneumonia Detection Report")

    c.setFont("Helvetica", 10)
    c.drawString(48, h - 74, f"Report ID: {report_id}")
    c.drawString(48, h - 88, f"Timestamp: {payload.get('timestamp', '')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, h - 114, f"Prediction: {payload.get('label', '—')}")
    c.setFont("Helvetica", 11)
    c.drawString(48, h - 132, f"Confidence: {payload.get('confidence', '—')}%")

    probs = payload.get("probabilities", {}) or {}
    y = h - 160
    c.setFont("Helvetica-Bold", 11)
    c.drawString(48, y, "Class probabilities:")
    y -= 14
    c.setFont("Helvetica", 10)
    for k, v in probs.items():
        c.drawString(60, y, f"- {k}: {v}%")
        y -= 12
        if y < 360:
            break

    # Images
    def _img_reader(b64: Optional[str]) -> Optional[ImageReader]:
        if not b64:
            return None
        try:
            raw = base64.b64decode(b64)
            return ImageReader(io.BytesIO(raw))
        except Exception:
            return None

    in_img = _img_reader(payload.get("img_b64"))
    cam_img = _img_reader(payload.get("gradcam_b64"))

    img_y = 70
    img_h = 240
    img_w = 240
    c.setFont("Helvetica-Bold", 11)
    c.drawString(48, img_y + img_h + 18, "Input")
    c.drawString(320, img_y + img_h + 18, "Grad-CAM")

    if in_img is not None:
        c.drawImage(in_img, 48, img_y, width=img_w, height=img_h, preserveAspectRatio=True, anchor="c")
    else:
        c.setFont("Helvetica", 10)
        c.drawString(48, img_y + img_h / 2, "No input image available")

    if cam_img is not None:
        c.drawImage(cam_img, 320, img_y, width=img_w, height=img_h, preserveAspectRatio=True, anchor="c")
    else:
        c.setFont("Helvetica", 10)
        c.drawString(320, img_y + img_h / 2, "Grad-CAM not available")

    c.showPage()
    c.save()

    pdf = buf.getvalue()
    resp = make_response(pdf)
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = f'attachment; filename="pneumonia_report_{report_id}.pdf"'
    return resp


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded. Field name must be 'image'."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload a PNG or JPEG."}), 400

    raw = file.read()
    if len(raw) > MAX_FILE_BYTES:
        return jsonify({"error": "File too large. Max size is 5 MB."}), 413

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return jsonify({"error": "Could not read image. The file may be corrupted."}), 400

    img_rgb = img.convert("RGB")
    img_array = preprocess(img_rgb)

    preds = MODEL.predict(img_array, verbose=0)
    preds = np.asarray(preds).reshape(-1)
    if preds.size == 0:
        return jsonify({"error": "Model returned empty prediction."}), 500

    class_idx = int(np.argmax(preds))
    label = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
    confidence = float(preds[class_idx]) * 100.0

    probabilities: Dict[str, float] = {}
    for i, p in enumerate(preds.tolist()):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"CLASS_{i}"
        probabilities[name] = round(float(p) * 100.0, 1)

    gradcam_img = _make_gradcam_overlay(img_rgb.resize(IMG_SIZE), img_array)
    gradcam_b64 = to_b64_png(gradcam_img) if gradcam_img is not None else None

    result = {
        "id": secrets.token_hex(4),
        "timestamp": _now_ts(),
        "label": label,
        "confidence": round(confidence, 1),
        "is_pneumonia": "PNEUMONIA" in label.upper(),
        "probabilities": probabilities,
        "img_b64": to_b64_png(img_rgb.resize(IMG_SIZE)),
        "gradcam_b64": gradcam_b64,
        "model_source": MODEL_SOURCE,
    }
    if "demo" in MODEL_SOURCE.lower() or "fallback" in MODEL_SOURCE.lower():
        result["note"] = (
            "Demo model loaded (fallback). Your trained model weights were not found/loaded, "
            "so predictions may be ~50/50. Train a model to create 'model.keras' or restore a complete SavedModel."
        )
    _report_cache_put(result["id"], result)

    history = session.get("history", [])
    history.insert(
        0,
        {
            "id": result["id"],
            "timestamp": result["timestamp"],
            "label": result["label"],
            "confidence": result["confidence"],
        },
    )
    session["history"] = history[:5]

    return jsonify(result)


if __name__ == "__main__":
    # Use FLASK_DEBUG=1 for debug mode.
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=debug)
