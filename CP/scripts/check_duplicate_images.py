from pathlib import Path
import hashlib


IMAGE_ROOT = Path("data/raw/fashioniq")


PAIRS = [
    ("dress", "B00C40W020", "B009RV9EAA"),
    ("dress", "B0091GQV3E", "B0091GQTNQ"),
]


def md5(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


for category, asin_a, asin_b in PAIRS:
    path_a = IMAGE_ROOT / category / f"{asin_a}.jpg"
    path_b = IMAGE_ROOT / category / f"{asin_b}.jpg"

    print()
    print("=" * 60)
    print(f"{asin_a} vs {asin_b}")

    print("A exists:", path_a.exists())
    print("B exists:", path_b.exists())

    if not path_a.exists() or not path_b.exists():
        continue

    print("A:", path_a)
    print("B:", path_b)

    hash_a = md5(path_a)
    hash_b = md5(path_b)

    print("MD5 A:", hash_a)
    print("MD5 B:", hash_b)
    print("Files identical:", hash_a == hash_b)