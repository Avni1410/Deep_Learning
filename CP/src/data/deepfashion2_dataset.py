from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DeepFashion2DomainDataset(Dataset):
    """
    Reusable DeepFashion2 consumer-to-shop paired dataset.

    Each sample contains:
        - consumer/user image
        - commercial/shop image
        - pair metadata

    Expected CSV columns:
        pair_id
        style
        category_name
        category_id
        user_image
        shop_image
        user_bbox
        shop_bbox
        user_path
        shop_path

    Expected image layout:

        image_root/
        +-- user/
        +-- shop/
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
            "pair_id",
            "style",
            "category_name",
            "category_id",
            "user_image",
            "shop_image",
        }

        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: "
                f"{sorted(missing_columns)}"
            )

    def __len__(self):
        return len(self.df)

    def _image_path(self, domain, filename):
        return self.image_root / domain / str(filename)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user_image_name = str(row["user_image"])
        shop_image_name = str(row["shop_image"])

        user_path = self._image_path("user", user_image_name)
        shop_path = self._image_path("shop", shop_image_name)

        if not user_path.exists():
            raise FileNotFoundError(
                f"User image not found: {user_path}"
            )

        if not shop_path.exists():
            raise FileNotFoundError(
                f"Shop image not found: {shop_path}"
            )

        user_image = Image.open(user_path).convert("RGB")
        shop_image = Image.open(shop_path).convert("RGB")

        if self.transform is not None:
            user_image = self.transform(user_image)
            shop_image = self.transform(shop_image)

        return {
            "user_image": user_image,
            "shop_image": shop_image,
            "pair_id": int(row["pair_id"]),
            "style": int(row["style"]),
            "category_name": str(row["category_name"]),
            "category_id": int(row["category_id"]),
            "user_image_name": user_image_name,
            "shop_image_name": shop_image_name,
            "user_path": str(user_path),
            "shop_path": str(shop_path),
        }
