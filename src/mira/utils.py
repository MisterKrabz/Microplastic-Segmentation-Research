"""Shared utilities for the MIRA framework."""

import os
import glob
import re

import numpy as np
import torch

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

YOLO_MODEL_PATTERN = re.compile(r"hunter-yolo-v([\d.]+)\.pt$")


def get_device(device: str | None = None) -> str:
    """Resolve the best available compute device."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def scan_images(*paths: str) -> list[str]:
    """
    Recursively collect image files from one or more directories.

    Accepts individual file paths, directory paths, or a single
    comma-separated string of paths.
    """
    image_paths: list[str] = []

    expanded: list[str] = []
    for p in paths:
        expanded.extend(part.strip() for part in p.split(",") if part.strip())

    for entry in expanded:
        if os.path.isfile(entry):
            if entry.lower().endswith(VALID_EXTS):
                image_paths.append(entry)
        elif os.path.isdir(entry):
            for root, _, files in os.walk(entry):
                for f in files:
                    if f.lower().endswith(VALID_EXTS):
                        image_paths.append(os.path.join(root, f))

    return sorted(set(image_paths))


def label_path_for_image(img_path: str) -> str:
    """
    Derive the YOLO label file path from an image path.

    Maps ``<prefix>/images/<name>.<ext>`` to ``<prefix>/labels/<name>.txt``.
    Falls back to swapping the extension to ``.txt`` if the path does not
    contain an ``images`` directory component.
    """
    p = os.path.normpath(img_path)
    parts = p.split(os.sep)

    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return os.path.splitext(os.sep.join(parts))[0] + ".txt"

    return os.path.splitext(p)[0] + ".txt"


def parse_yolo_seg_labels(label_path: str, img_w: int, img_h: int) -> list[np.ndarray]:
    """Parse a YOLO segmentation label file into a list of binary masks."""
    import cv2

    masks: list[np.ndarray] = []
    if not os.path.exists(label_path):
        return masks

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue

            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)
                points.append([x, y])

            if len(points) >= 3:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)

    return masks


def count_gt_instances(label_path: str) -> int:
    """Count the number of ground-truth instances in a label file."""
    if not os.path.exists(label_path):
        return 0
    n = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def discover_yolo_versions(models_dir: str) -> dict[str, str]:
    """
    Scan a directory for YOLO model files matching the naming convention
    ``hunter-yolo-v<version>.pt`` and return a ``{version: path}`` mapping.
    """
    versions: dict[str, str] = {}
    if not os.path.isdir(models_dir):
        return versions

    for entry in os.listdir(models_dir):
        m = YOLO_MODEL_PATTERN.match(entry)
        if m:
            versions[m.group(1)] = os.path.join(models_dir, entry)

    return versions


def resolve_yolo_model(
    model_path: str | None = None,
    *,
    version: str | None = None,
    models_dir: str | None = None,
) -> str:
    """
    Resolve a YOLO model checkpoint path.

    Either supply ``model_path`` directly, or provide a ``version`` string
    (e.g. ``"0.4.4"``) together with ``models_dir`` to auto-discover it.
    """
    if model_path is not None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        return model_path

    if version is None or models_dir is None:
        raise ValueError(
            "Provide either model_path, or both version and models_dir."
        )

    version = version.lstrip("v")
    available = discover_yolo_versions(models_dir)

    if version not in available:
        raise FileNotFoundError(
            f"YOLO version v{version} not found in {models_dir}. "
            f"Available: {sorted(available.keys()) or 'none'}"
        )

    return available[version]
