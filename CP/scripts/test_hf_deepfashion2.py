from huggingface_hub import list_repo_files

REPO_ID = "sahirp/deepfashion2"

print("=" * 70)
print("Hugging Face DeepFashion2 repository test")
print("=" * 70)

try:
    files = list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset"
    )

    print(f"\nRepository: {REPO_ID}")
    print(f"Number of files/directories returned: {len(files)}")

    print("\nFirst 50 entries:\n")

    for item in files[:50]:
        print(item)

except Exception as e:
    print("\nERROR:")
    print(type(e).__name__)
    print(e)