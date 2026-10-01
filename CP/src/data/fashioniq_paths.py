"""
Phase 6 - centralized FashionIQ image path resolution, matching
FashionIQCIRDataset._image_path() exactly: data/raw/fashioniq/<category>/<ASIN>.jpg

Used by train_phase6.py to re-encode mined hard-negative images by ASIN
(FashionIQCIRDataset itself only resolves reference/target pairs from its
own CSV rows, not arbitrary negative IDs, so this does not duplicate its
logic - it mirrors the one existing convention for a different access pattern).
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
            "Not silently skipped, per project policy - this must be investigated, "
            "not papered over, since it would silently shrink the negative count."
        )
    return Image.open(path).convert("RGB")