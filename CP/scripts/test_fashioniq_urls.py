from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "fashioniq"

TEST_IDS = {
    "B00D7MZJEA",
    "B008LR38J4",
    "B00EEGIN3W",
    "B00CIPJ4L6",
    "B00B2IMVBK",
    "B00DY81FLS",
}


def load_urls():
    results = {}

    for path in DATA.glob("asin2url.*.txt"):

        for line in path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines():

            parts = line.split()

            if len(parts) >= 2:

                image_id = parts[0]
                url = parts[1]

                if image_id in TEST_IDS:
                    results[image_id] = url

    return results


def test_url(image_id, url):

    print()
    print("-" * 70)
    print(image_id)
    print("Original:", url)

    candidates = [
        url,
        url.replace("http://", "https://"),
    ]

    headers_list = [
        {},
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0 Safari/537.36"
            )
        },
    ]

    for candidate in candidates:

        for headers in headers_list:

            try:

                response = requests.get(
                    candidate,
                    headers=headers,
                    timeout=20,
                    allow_redirects=True,
                )

                print(
                    "URL:",
                    candidate,
                    "| status:",
                    response.status_code,
                    "| bytes:",
                    len(response.content),
                    "| content-type:",
                    response.headers.get(
                        "Content-Type"
                    ),
                )

                if response.ok:

                    try:

                        image = Image.open(
                            BytesIO(response.content)
                        )

                        image.verify()

                        print(
                            "VALID IMAGE:",
                            image_id,
                            "| format:",
                            image.format,
                        )

                        return True

                    except Exception as e:

                        print(
                            "Response is not a valid image:",
                            e
                        )

            except Exception as e:

                print(
                    "REQUEST ERROR:",
                    type(e).__name__,
                    str(e)
                )

    return False


urls = load_urls()

print("=" * 70)
print("FASHIONIQ URL RECOVERY TEST")
print("=" * 70)

successful = 0

for image_id in sorted(TEST_IDS):

    url = urls.get(image_id)

    if not url:

        print(image_id, "URL NOT FOUND")
        continue

    if test_url(image_id, url):
        successful += 1


print()
print("=" * 70)
print(
    f"Successful recovery tests: "
    f"{successful}/{len(TEST_IDS)}"
)
print("=" * 70)