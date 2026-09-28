from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deepfashion2"
    / "domain_alignment_pairs_5000.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "scripts"

df = pd.read_csv(CSV_PATH)

# Get unique user/consumer images
user_files = sorted(
    f"train/image/{name}"
    for name in df["user_image"].dropna().unique()
)

# Get unique shop/commercial images
shop_files = sorted(
    f"train/image/{name}"
    for name in df["shop_image"].dropna().unique()
)

# Save file lists
user_list = OUTPUT_DIR / "deepfashion2_user_files.txt"
shop_list = OUTPUT_DIR / "deepfashion2_shop_files.txt"

user_list.write_text("\n".join(user_files), encoding="utf-8")
shop_list.write_text("\n".join(shop_files), encoding="utf-8")

print(f"User images: {len(user_files)}")
print(f"Shop images: {len(shop_files)}")
print(f"Total unique images: {len(set(user_files) | set(shop_files))}")

print("\nCreated:")
print(user_list)
print(shop_list)