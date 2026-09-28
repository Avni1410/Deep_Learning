import zipfile
from pathlib import Path

import pandas as pd


ZIP_PATH = Path("data/raw/deepfashion2/train.zip")
CSV_PATH = Path("data/processed/deepfashion2/domain_alignment_pairs_5000.csv")
OUTPUT_ROOT = Path("data/processed/deepfashion2/images")


def main():
    print("=" * 70)
    print("DeepFashion2 Subset Extraction")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load manifest
    # ---------------------------------------------------------
    print("\nLoading pair manifest...")

    df = pd.read_csv(CSV_PATH)

    user_images = set(df["user_image"].dropna())
    shop_images = set(df["shop_image"].dropna())
    all_images = user_images | shop_images

    print(f"Pairs:              {len(df):,}")
    print(f"Unique user images: {len(user_images):,}")
    print(f"Unique shop images: {len(shop_images):,}")
    print(f"Unique total:       {len(all_images):,}")

    # ---------------------------------------------------------
    # 2. Create output directories
    # ---------------------------------------------------------
    user_dir = OUTPUT_ROOT / "user"
    shop_dir = OUTPUT_ROOT / "shop"

    user_dir.mkdir(parents=True, exist_ok=True)
    shop_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 3. Open ZIP
    # ---------------------------------------------------------
    print("\nOpening DeepFashion2 train.zip...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        print("ZIP opened successfully.")

        # -----------------------------------------------------
        # 4. Build lookup of files inside ZIP
        # -----------------------------------------------------
        print("\nReading ZIP file list...")

        zip_names = set(z.namelist())

        print(f"Files in ZIP: {len(zip_names):,}")

        # -----------------------------------------------------
        # 5. Verify requested images exist
        # -----------------------------------------------------
        missing = []

        for image_name in all_images:
            zip_path = f"train/image/{image_name}"

            if zip_path not in zip_names:
                missing.append(image_name)

        print(f"\nRequested images: {len(all_images):,}")
        print(f"Missing images:    {len(missing):,}")

        if missing:
            print("\nFirst missing images:")

            for image in missing[:20]:
                print(f"  {image}")

            raise RuntimeError(
                f"{len(missing)} requested images were not found "
                "inside train.zip."
            )

        print("All requested images exist in the ZIP.")

        # -----------------------------------------------------
        # 6. Extract user images
        # -----------------------------------------------------
        print("\nExtracting user images...")

        for index, image_name in enumerate(
            sorted(user_images), start=1
        ):
            source = f"train/image/{image_name}"
            destination = user_dir / image_name

            if not destination.exists():
                with z.open(source) as src, open(
                    destination, "wb"
                ) as dst:
                    dst.write(src.read())

            if index % 500 == 0 or index == len(user_images):
                print(
                    f"  User images: "
                    f"{index:,}/{len(user_images):,}"
                )

        # -----------------------------------------------------
        # 7. Extract shop images
        # -----------------------------------------------------
        print("\nExtracting shop images...")

        for index, image_name in enumerate(
            sorted(shop_images), start=1
        ):
            source = f"train/image/{image_name}"
            destination = shop_dir / image_name

            if not destination.exists():
                with z.open(source) as src, open(
                    destination, "wb"
                ) as dst:
                    dst.write(src.read())

            if index % 500 == 0 or index == len(shop_images):
                print(
                    f"  Shop images: "
                    f"{index:,}/{len(shop_images):,}"
                )

    # ---------------------------------------------------------
    # 8. Final verification
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("Final Verification")
    print("=" * 70)

    extracted_user = {
        p.name for p in user_dir.glob("*.jpg")
    }

    extracted_shop = {
        p.name for p in shop_dir.glob("*.jpg")
    }

    missing_user = user_images - extracted_user
    missing_shop = shop_images - extracted_shop

    print(f"\nExpected user images:    {len(user_images):,}")
    print(f"Extracted user images:   {len(extracted_user):,}")

    print(f"\nExpected shop images:    {len(shop_images):,}")
    print(f"Extracted shop images:   {len(extracted_shop):,}")

    print(f"\nMissing user images:     {len(missing_user):,}")
    print(f"Missing shop images:     {len(missing_shop):,}")

    if missing_user or missing_shop:
        print("\n❌ Extraction verification FAILED.")

        if missing_user:
            print("\nMissing user images:")
            for image in sorted(missing_user)[:20]:
                print(f"  {image}")

        if missing_shop:
            print("\nMissing shop images:")
            for image in sorted(missing_shop)[:20]:
                print(f"  {image}")

        raise RuntimeError("Some images were not extracted.")

    print("\n✅ All requested images extracted successfully.")

    print("\nOutput:")
    print(f"  {user_dir}")
    print(f"  {shop_dir}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()