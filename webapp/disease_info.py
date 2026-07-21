"""
disease_info.py
Helper module to load and serve plant disease information.
Reads from disease_info.json in the same directory.
"""

import json
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DISEASE_DB_PATH = os.path.join(_BASE_DIR, "disease_info.json")

# Load once at import time
with open(_DISEASE_DB_PATH, "r") as f:
    _DISEASE_DB: dict = json.load(f)


def get_disease_info(class_name: str) -> dict:
    """
    Return disease information for a given class name.
    Falls back to a generic entry if the class is not in the database.

    Args:
        class_name: The raw class name from the model's label mapping
                    (e.g. 'Apple___Apple_scab')

    Returns:
        dict with keys: display_name, plant, status, severity, color,
                        description, symptoms, causes, treatment, prevention
    """
    info = _DISEASE_DB.get(class_name)
    if info is not None:
        return info

    # Graceful fallback — parse class name for a best-effort response
    parts = class_name.replace("___", " — ").replace("_", " ")
    is_healthy = "healthy" in class_name.lower()
    return {
        "display_name": parts,
        "plant": class_name.split("___")[0].replace("_", " ") if "___" in class_name else class_name,
        "status": "healthy" if is_healthy else "diseased",
        "severity": "none" if is_healthy else "unknown",
        "color": "#22c55e" if is_healthy else "#f59e0b",
        "description": "Information about this class is not yet available in the database.",
        "symptoms": [],
        "causes": [],
        "treatment": [],
        "prevention": [],
    }


def get_all_classes() -> list[str]:
    """Return all class names in the disease database."""
    return list(_DISEASE_DB.keys())


def get_display_name(class_name: str) -> str:
    """Return human-friendly display name for a class."""
    info = _DISEASE_DB.get(class_name)
    if info:
        return info["display_name"]
    return class_name.replace("___", " — ").replace("_", " ")


def is_healthy(class_name: str) -> bool:
    """Return True if the class represents a healthy plant."""
    info = _DISEASE_DB.get(class_name)
    if info:
        return info["status"] == "healthy"
    return "healthy" in class_name.lower()
