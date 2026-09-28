from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FashionIQCIRDataset(Dataset):
    """
    Reusable FashionIQ dataset for composed image retrieval.

    Each sample contains:
        - reference image (candidate)
        - target image
        - two modification captions
        - category/split/query metadata

    Expected CSV columns:
        category
        split
        query_id
        candidate
        target
        caption_1
        caption_2

    Expected image layout:

        image_root/
        +-- dress/
        +-- shirt/
        +-- toptee/

    Image filenames are expected to be:
        <ASIN>.jpg
    """

    def __init__(
        self,
        csv_path,
        image_root,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform

        self.df = pd.read_csv(self.csv_path)

        required_columns = {
            "category",
            "split",
            "query_id",
            "candidate",
            "target",
            "caption_1",
            "caption_2",
        }

        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: "
                f"{sorted(missing_columns)}"
            )

    def __len__(self):
        return len(self.df)

    def _image_path(self, category, asin):
        return self.image_root / category / f"{asin}.jpg"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        category = str(row["category"])
        candidate = str(row["candidate"])
        target = str(row["target"])

        reference_path = self._image_path(category, candidate)
        target_path = self._image_path(category, target)

        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference image not found: {reference_path}"
            )

        if not target_path.exists():
            raise FileNotFoundError(
                f"Target image not found: {target_path}"
            )

        reference_image = Image.open(reference_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform is not None:
            reference_image = self.transform(reference_image)
            target_image = self.transform(target_image)

        return {
            "reference_image": reference_image,
            "target_image": target_image,
            "caption_1": str(row["caption_1"]),
            "caption_2": str(row["caption_2"]),
            "category": category,
            "split": str(row["split"]),
            "query_id": str(row["query_id"]),
            "candidate": candidate,
            "target": target,
            "reference_path": str(reference_path),
            "target_path": str(target_path),
        }
