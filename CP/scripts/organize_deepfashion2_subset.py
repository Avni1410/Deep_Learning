from pathlib import Path
import pandas as pd
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deepfashion2"
    / "domain_alignment_pairs_5000.csv"
)

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deepfashion2"
    / "temp_train"
    / "image"
)

USER_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deepfashion2"
    / "images"
    / "user"
)

SHOP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deepfashion2"
    / "images"
    / "shop"
)

df = pd.read_csv(CSV_PATH)

user_images = sorted(df["user_image"].dropna().unique())
shop_images = sorted(df["shop_image"].dropna().unique())

print(f"User images required: {len(user_images)}")
print(f"Shop images required: {len(shop_images)}")

missing_user = []
missing_shop = []

print("\nCopying user images...")

for name in user_images:
    src = SOURCE_DIR / name
    dst = USER_DIR / name

    if src.exists():
        shutil.copy2(src, dst)
    else:
        missing_user.append(name)

print("Copying shop images...")

for name in shop_images:
    src = SOURCE_DIR / name
    dst = SHOP_DIR / name

    if src.exists():
        shutil.copy2(src, dst)
    else:
        missing_shop.append(name)

print("\n========== RESULT ==========")
print(f"User copied: {len(user_images) - len(missing_user)}")
print(f"User missing: {len(missing_user)}")
print(f"Shop copied: {len(shop_images) - len(missing_shop)}")
print(f"Shop missing: {len(missing_shop)}")

if missing_user:
    print("\nMissing user images:")
    print(missing_user[:20])

if missing_shop:
    print("\nMissing shop images:")
    print(missing_shop[:20])