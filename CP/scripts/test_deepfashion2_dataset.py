from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DeepFashion2DomainDataset(Dataset):
    def __init__(self, csv_path, image_root):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user_path = self.image_root / "user" / row["user_image"]
        shop_path = self.image_root / "shop" / row["shop_image"]

        user_image = Image.open(user_path).convert("RGB")
        shop_image = Image.open(shop_path).convert("RGB")

        return {
            "user_image": user_image,
            "shop_image": shop_image,
            "pair_id": int(row["pair_id"]),
            "style": int(row["style"]),
            "category_name": row["category_name"],
            "category_id": int(row["category_id"]),
            "user_image_name": row["user_image"],
            "shop_image_name": row["shop_image"],
        }


def collate_fn(batch):
    """
    Keep PIL images as a list for now.

    We will convert them into tensors using the
    CLIP preprocessing pipeline later.
    """

    return {
        "user_images": [item["user_image"] for item in batch],
        "shop_images": [item["shop_image"] for item in batch],
        "pair_ids": [item["pair_id"] for item in batch],
        "styles": [item["style"] for item in batch],
        "categories": [item["category_name"] for item in batch],
        "category_ids": [item["category_id"] for item in batch],
        "user_image_names": [item["user_image_name"] for item in batch],
        "shop_image_names": [item["shop_image_name"] for item in batch],
    }


CSV_PATH = Path(
    "data/processed/deepfashion2/domain_alignment_pairs_5000.csv"
)

IMAGE_ROOT = Path(
    "data/processed/deepfashion2/images"
)


dataset = DeepFashion2DomainDataset(
    csv_path=CSV_PATH,
    image_root=IMAGE_ROOT,
)


loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)


print("=" * 60)
print("DeepFashion2 DataLoader Test")
print("=" * 60)

print(f"Dataset size: {len(dataset)}")
print(f"Batch size:   4")


batch = next(iter(loader))


print("\nBatch information:")
print(f"Number of user images: {len(batch['user_images'])}")
print(f"Number of shop images: {len(batch['shop_images'])}")

print("\nPair IDs:")
print(batch["pair_ids"])

print("\nCategories:")
print(batch["categories"])

print("\nUser image names:")
print(batch["user_image_names"])

print("\nShop image names:")
print(batch["shop_image_names"])

print("\nImage sizes:")

for i in range(len(batch["user_images"])):
    print(
        f"Pair {batch['pair_ids'][i]} | "
        f"user={batch['user_images'][i].size} | "
        f"shop={batch['shop_images'][i].size}"
    )

print("\nPair integrity:")

for i in range(len(batch["pair_ids"])):
    print(
        f"Pair ID {batch['pair_ids'][i]}: "
        f"{batch['user_image_names'][i]} -> "
        f"{batch['shop_image_names'][i]}"
    )

print("=" * 60)
print("DataLoader test completed successfully.")
print("=" * 60)