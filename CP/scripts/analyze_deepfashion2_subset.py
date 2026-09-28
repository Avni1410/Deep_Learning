import pandas as pd
from pathlib import Path

INPUT = Path(
    "data/processed/deepfashion2/domain_alignment_pairs.csv"
)

df = pd.read_csv(INPUT)

user_images = set(df["user_image"])
shop_images = set(df["shop_image"])

all_images = user_images | shop_images

print("=" * 70)
print("DeepFashion2 Selected Subset Analysis")
print("=" * 70)

print(f"\nPairs:              {len(df):,}")
print(f"Unique user images: {len(user_images):,}")
print(f"Unique shop images: {len(shop_images):,}")
print(f"Unique all images:  {len(all_images):,}")

overlap = user_images & shop_images

print(f"User/shop overlap:  {len(overlap):,}")

print("\nCategory counts:")
print(
    df["category_name"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "=" * 70)