from pathlib import Path
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

CLASS_NAMES = ["No Pain", "Low Pain", "Moderate Pain", "High Pain"]

def build_model(num_classes=4):
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)
    return model

def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError("Unsupported checkpoint format. Expected a state_dict or a training checkpoint containing one.")

def load_model(checkpoint_path):
    model = build_model()
    if not checkpoint_path.exists():
        return None, f"Checkpoint not found: {checkpoint_path}"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = _extract_state_dict(checkpoint)
        cleaned = {}
        for k, v in state.items():
            k = k.replace("module.", "")
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            return None, f"Checkpoint loaded with missing keys: {missing[:5]}"
        model.eval()
        return model, f"Loaded checkpoint: {checkpoint_path}"
    except Exception as e:
        return None, f"Checkpoint load failed: {type(e).__name__}: {e}"
