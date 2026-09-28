import json
from pathlib import Path
from collections import Counter

path = Path("data/raw/deepfashion2/matching_pairs.json")

print("=" * 70)
print("DeepFashion2 matching_pairs.json inspection")
print("=" * 70)

if not path.exists():
    print(f"ERROR: File not found: {path}")
    raise SystemExit(1)

print(f"\nFile: {path}")
print(f"Size: {path.stat().st_size / (1024**2):.2f} MB")

print("\nLoading JSON...")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Root type: {type(data).__name__}")

if isinstance(data, dict):
    print(f"Number of root keys: {len(data)}")
    print("\nFirst 10 root keys:")

    for key in list(data.keys())[:10]:
        print(" ", repr(key))

elif isinstance(data, list):
    print(f"Number of list elements: {len(data)}")

print("\n" + "=" * 70)
print("First element / sample")
print("=" * 70)

if isinstance(data, dict):
    first_key = next(iter(data))
    print(f"First key: {first_key}")
    print("Value:")
    print(data[first_key])

elif isinstance(data, list) and len(data) > 0:
    print(data[0])

print("\n" + "=" * 70)
print("Key structure analysis")
print("=" * 70)

counter = Counter()

def inspect(obj, prefix="root"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            counter[f"{prefix}.{key}"] += 1
            if isinstance(value, dict):
                inspect(value, f"{prefix}.{key}")
            elif isinstance(value, list) and value:
                inspect(value[0], f"{prefix}.{key}[]")

    elif isinstance(obj, list) and obj:
        inspect(obj[0], f"{prefix}[]")

inspect(data)

for key, count in counter.most_common():
    print(f"{key}: {count}")

print("\n" + "=" * 70)
print("Inspection complete")
print("=" * 70)