"""
inference.py
ViT inference pipeline for PhytoGuard web application.

Loads the fine-tuned ViTForImageClassification model from:
    ../saved_models/vit_base_patch16_final/

Uses the exact same preprocessing as during training:
    - Resize to 224x224 (ViTImageProcessor)
    - Normalize: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

Label mapping loaded from:
    ../saved_models/vit_base_label_mappings.json
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import ViTForImageClassification, ViTImageProcessor

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
_PROJECT_ROOT = _BASE_DIR.parent
_MODEL_DIR = _PROJECT_ROOT / "saved_models" / "vit_base_patch16_final"
_LABEL_MAP_PATH = _PROJECT_ROOT / "saved_models" / "vit_base_label_mappings.json"
_CLASSIFICATION_REPORT_PATH = _PROJECT_ROOT / "saved_models" / "vit_base_classification_report.json"

# ── Constants (from preprocessor_config.json) ──────────────────────────────────
IMAGE_SIZE = 224
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]
NUM_CLASSES = 11

# ── Image validation thresholds ─────────────────────────────────────────────────
MIN_IMAGE_SIZE_PX = 100          # minimum width or height in pixels
BLUR_THRESHOLD = 50.0            # Laplacian variance; below this = blurry


# ── Singleton model holder ──────────────────────────────────────────────────────
class _ModelHolder:
    model: Optional[ViTForImageClassification] = None
    processor: Optional[ViTImageProcessor] = None
    id2label: Optional[dict] = None
    device: Optional[torch.device] = None
    model_size_mb: float = 0.0
    accuracy: float = 0.9998


_holder = _ModelHolder()


def _get_device() -> torch.device:
    """Select best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model() -> None:
    """
    Load the ViT model and processor into memory (called once at startup).
    Raises RuntimeError if model files are missing.
    """
    if _holder.model is not None:
        return  # already loaded

    if not _MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found: {_MODEL_DIR}\n"
            "Please ensure saved_models/vit_base_patch16_final/ exists in the project root."
        )

    logger.info("Loading ViT model from %s ...", _MODEL_DIR)

    # Load label mapping
    with open(_LABEL_MAP_PATH, "r") as f:
        label_data = json.load(f)
    _holder.id2label = {int(k): v for k, v in label_data["id2label"].items()}

    # Select device
    _holder.device = _get_device()
    logger.info("Using device: %s", _holder.device)

    # Load processor and model
    _holder.processor = ViTImageProcessor.from_pretrained(str(_MODEL_DIR))
    _holder.model = ViTForImageClassification.from_pretrained(str(_MODEL_DIR))
    _holder.model.to(_holder.device)
    _holder.model.eval()

    # Calculate model size
    safetensors_path = _MODEL_DIR / "model.safetensors"
    if safetensors_path.exists():
        _holder.model_size_mb = safetensors_path.stat().st_size / (1024 * 1024)

    # Load accuracy from classification report
    if _CLASSIFICATION_REPORT_PATH.exists():
        with open(_CLASSIFICATION_REPORT_PATH, "r") as f:
            report = json.load(f)
        _holder.accuracy = report.get("accuracy", 0.9998)

    logger.info(
        "Model loaded successfully. Size: %.1f MB, Device: %s",
        _holder.model_size_mb,
        _holder.device,
    )


# ── Image validation ────────────────────────────────────────────────────────────

def validate_image(image: Image.Image) -> tuple[bool, str]:
    """
    Validate the uploaded image before running inference.

    Returns:
        (is_valid: bool, message: str)
    """
    import cv2

    # 1. Minimum size check
    w, h = image.size
    if w < MIN_IMAGE_SIZE_PX or h < MIN_IMAGE_SIZE_PX:
        return False, (
            f"Image is too small ({w}×{h} px). "
            f"Please upload an image at least {MIN_IMAGE_SIZE_PX}×{MIN_IMAGE_SIZE_PX} pixels."
        )

    # 2. Blur detection using Laplacian variance
    try:
        img_array = np.array(image.convert("L"))  # convert to grayscale
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        if laplacian_var < BLUR_THRESHOLD:
            return False, (
                f"Image appears blurry (sharpness score: {laplacian_var:.1f}). "
                "Please upload a clearer, in-focus image."
            )
    except Exception:
        pass  # if cv2 fails, skip blur check gracefully

    return True, "OK"


# ── Inference ───────────────────────────────────────────────────────────────────

def predict(image: Image.Image) -> dict:
    """
    Run ViT inference on a PIL image.

    Args:
        image: PIL.Image.Image (RGB)

    Returns:
        dict with keys:
            - predicted_class: str
            - confidence: float (0–100)
            - top3: list of {class_name, confidence} dicts
            - inference_time_ms: float
            - device: str
    """
    if _holder.model is None:
        load_model()

    # Ensure RGB
    image = image.convert("RGB")

    # Preprocess — uses saved ViTImageProcessor (224×224, mean/std=0.5)
    inputs = _holder.processor(images=image, return_tensors="pt")
    inputs = {k: v.to(_holder.device) for k, v in inputs.items()}

    # Inference with timing
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = _holder.model(**inputs)
    t1 = time.perf_counter()

    inference_time_ms = (t1 - t0) * 1000.0

    # Convert logits to probabilities
    logits = outputs.logits  # shape: (1, num_classes)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (num_classes,)

    # Top-1 prediction
    top1_idx = int(np.argmax(probs))
    top1_conf = float(probs[top1_idx]) * 100.0
    top1_class = _holder.id2label[top1_idx]

    # Top-3 predictions
    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "class_name": _holder.id2label[int(i)],
            "confidence": float(probs[i]) * 100.0,
        }
        for i in top3_indices
    ]

    return {
        "predicted_class": top1_class,
        "confidence": round(top1_conf, 2),
        "top3": top3,
        "inference_time_ms": round(inference_time_ms, 1),
        "device": str(_holder.device),
    }


def get_model_info() -> dict:
    """Return model metadata for the UI model information panel."""
    if _holder.model is None:
        load_model()

    return {
        "architecture": "Vision Transformer (ViT-Base/16)",
        "framework": "PyTorch + HuggingFace Transformers",
        "model_type": "ViTForImageClassification",
        "num_classes": NUM_CLASSES,
        "input_size": f"{IMAGE_SIZE}×{IMAGE_SIZE} pixels",
        "patch_size": 16,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "normalization": f"mean={IMAGE_MEAN}, std={IMAGE_STD}",
        "model_size_mb": round(_holder.model_size_mb, 1),
        "device": str(_holder.device),
        "validation_accuracy": f"{_holder.accuracy * 100:.2f}%",
        "training_epochs": 10,
        "dataset": "PlantVillage (subset — 11 classes)",
        "train_images": 20789,
        "val_images": 5198,
    }
