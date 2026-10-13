"""
Phase 8 - centralized FashionIQ image path resolution, matching
FashionIQCIRDataset's own convention exactly:
    data/raw/fashioniq/<category>/<ASIN>.jpg

Used to load mined hard-negative images by (category, ASIN), a lookup
FashionIQCIRDataset itself does not expose (it only resolves the
reference/target pair from its own CSV rows).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def fashioniq_image_path(image_root: Path, category: str, asin: str) -> Path:
    return Path(image_root) / category / f"{asin}.jpg"


def load_fashioniq_image(image_root: Path, category: str, asin: str) -> Image.Image:
    path = fashioniq_image_path(image_root, category, asin)
    if not path.exists():
        raise FileNotFoundError(
            f"Hard-negative image not found: {path} (category={category}, asin={asin}). "
            "Not silently skipped - would make the mined-negative count wrong unnoticed."
        )
    return Image.open(path).convert("RGB")
