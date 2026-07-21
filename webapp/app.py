"""
app.py — PhytoGuard AI Web Application (Flask Backend)
=====================================================
Routes:
  GET  /                          → Serve main SPA
  POST /predict                   → Image upload + ViT inference → JSON
  GET  /model-info                → Model metadata JSON
  GET  /history                   → Session prediction history JSON
  GET  /samples                   → List of sample images from test/
  GET  /sample-image/<filename>   → Serve a sample image file

Launch:
  source ../m4_tensorflow/bin/activate
  python app.py
"""

import io
import json
import logging
import os
import sys
import base64
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, abort

# ── Ensure project root is on the path so inference.py can import ──────────────
_BASE_DIR = Path(__file__).parent
_PROJECT_ROOT = _BASE_DIR.parent
sys.path.insert(0, str(_BASE_DIR))

from inference import load_model, predict, validate_image, get_model_info
from disease_info import get_disease_info, get_display_name, is_healthy

from PIL import Image

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phytoguard")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "phytoguard-secret-2026")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ── In-memory session history ──────────────────────────────────────────────────
_history: list[dict] = []
MAX_HISTORY = 20

# ── Sample images directory ────────────────────────────────────────────────────
_TEST_DIR = _PROJECT_ROOT / "test"
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _pil_to_base64(img: Image.Image, max_size: int = 200) -> str:
    """Resize and encode PIL image to base64 data URL for history thumbnails."""
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """Accept an uploaded image and return inference results as JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        return jsonify({"error": "Unsupported file type. Please upload JPG or PNG."}), 400

    try:
        # Read image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to open uploaded image: %s", e)
        return jsonify({"error": "Could not read the image file. Is it a valid image?"}), 400

    # Validate image
    valid, msg = validate_image(image)
    if not valid:
        return jsonify({"error": msg, "validation_failed": True}), 422

    # Run inference
    try:
        result = predict(image)
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    # Enrich with disease info
    class_name = result["predicted_class"]
    disease_info = get_disease_info(class_name)

    # Build response
    response = {
        "predicted_class": class_name,
        "display_name": disease_info["display_name"],
        "plant": disease_info["plant"],
        "is_healthy": is_healthy(class_name),
        "confidence": result["confidence"],
        "top3": [
            {
                **item,
                "display_name": get_display_name(item["class_name"]),
                "is_healthy": is_healthy(item["class_name"]),
            }
            for item in result["top3"]
        ],
        "inference_time_ms": result["inference_time_ms"],
        "device": result["device"],
        "disease_info": disease_info,
        "timestamp": datetime.now().isoformat(),
    }

    # Add to history
    thumbnail = _pil_to_base64(image.copy())
    history_entry = {
        "id": str(uuid.uuid4())[:8],
        "thumbnail": thumbnail,
        "predicted_class": class_name,
        "display_name": disease_info["display_name"],
        "plant": disease_info["plant"],
        "is_healthy": is_healthy(class_name),
        "confidence": result["confidence"],
        "severity": disease_info.get("severity", "unknown"),
        "color": disease_info.get("color", "#6b7280"),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    _history.insert(0, history_entry)
    if len(_history) > MAX_HISTORY:
        _history.pop()

    return jsonify(response)


@app.route("/model-info")
def model_info_route():
    """Return model metadata."""
    try:
        info = get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.exception("Failed to get model info: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def history_route():
    """Return session prediction history."""
    return jsonify(_history)


@app.route("/samples")
def samples_route():
    """Return list of sample image filenames from the test directory."""
    if not _TEST_DIR.exists():
        return jsonify([])
    files = [
        f.name
        for f in sorted(_TEST_DIR.iterdir())
        if f.suffix in _ALLOWED_EXTENSIONS
    ]
    return jsonify(files)


@app.route("/sample-image/<filename>")
def sample_image_route(filename: str):
    """Serve a sample image from the test/ directory."""
    # Security: only allow simple filenames (no path traversal)
    if "/" in filename or "\\" in filename or ".." in filename:
        abort(403)
    image_path = _TEST_DIR / filename
    if not image_path.exists():
        abort(404)
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return send_file(str(image_path), mimetype=mime)


# ── Startup ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  PhytoGuard AI — Plant Disease Detection")
    logger.info("=" * 60)
    logger.info("Loading ViT model (this may take 10–20 seconds)...")
    try:
        load_model()
        logger.info("Model loaded successfully ✓")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.error("Please ensure the m4_tensorflow environment is activated.")
        sys.exit(1)

    logger.info("Starting Flask server on http://localhost:5001")
    logger.info("Open your browser and navigate to: http://localhost:5001")
    logger.info("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        threaded=True,
    )
