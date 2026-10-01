from pathlib import Path
import pandas as pd
import json
import html
import webbrowser


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

HARD_NEGATIVE_ROOT = (
    REPO_ROOT / "experiments" / "phase6" / "hard_negatives"
)

IMAGE_ROOT = (
    REPO_ROOT / "data" / "raw" / "fashioniq"
)

CAPTION_ROOT = (
    IMAGE_ROOT / "captions"
)

OUTPUT_ROOT = (
    REPO_ROOT / "experiments" / "phase6" / "visual_inspection"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CAPTIONS
# ============================================================

def load_captions(category):
    """
    Load FashionIQ train captions and create:
        candidate ASIN -> captions
    """

    caption_file = (
        CAPTION_ROOT / f"cap.{category}.train.json"
    )

    if not caption_file.exists():
        print(f"[WARNING] Caption file not found: {caption_file}")
        return {}

    with open(caption_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    caption_map = {}

    for item in data:
        candidate = item.get("candidate")

        if candidate is None:
            continue

        captions = item.get("captions", [])

        if candidate not in caption_map:
            caption_map[candidate] = captions

    return caption_map


# ============================================================
# IMAGE PATH
# ============================================================

def image_path(asin, category):
    """
    FashionIQ image path:
        data/raw/fashioniq/<category>/<ASIN>.jpg
    """

    path = IMAGE_ROOT / category / f"{asin}.jpg"

    if path.exists():
        return path

    return None


# ============================================================
# HTML HELPERS
# ============================================================

def image_html(path, alt):
    if path is None:
        return (
            '<div class="missing">IMAGE NOT FOUND</div>'
        )

    # Convert to relative path from output HTML directory
    relative = path.relative_to(OUTPUT_ROOT.parent.parent.parent)

    # HTML needs forward slashes
    relative = relative.as_posix()

    return f"""
    <img
        src="../../../{html.escape(relative)}"
        alt="{html.escape(alt)}"
        loading="lazy"
    >
    """


def card(title, asin, category, similarity=None):
    path = image_path(asin, category)

    similarity_text = ""

    if similarity is not None:
        similarity_text = (
            f'<div class="similarity">'
            f'similarity = {similarity:.4f}'
            f'</div>'
        )

    category_text = (
        f'<div class="category">{html.escape(category)}</div>'
    )

    return f"""
    <div class="card">
        <h3>{html.escape(title)}</h3>

        {image_html(path, asin)}

        <div class="asin">
            {html.escape(asin)}
        </div>

        {category_text}

        {similarity_text}
    </div>
    """


# ============================================================
# PROCESS ONE CATEGORY
# ============================================================

def build_category_html(category):

    csv_path = (
        HARD_NEGATIVE_ROOT
        / f"{category}_train_similarity_negatives.csv"
    )

    if not csv_path.exists():
        print(f"[WARNING] Missing CSV: {csv_path}")
        return ""

    df = pd.read_csv(csv_path)

    caption_map = load_captions(category)

    sections = []

    for _, row in df.iterrows():

        query_id = row["query_id"]

        target = row["target"]
        candidate = row["candidate"]

        neg1 = row["negative_1"]
        neg2 = row["negative_2"]
        neg3 = row["negative_3"]

        neg1_cat = row["negative_1_category"]
        neg2_cat = row["negative_2_category"]
        neg3_cat = row["negative_3_category"]

        sim_pos = float(row["positive_similarity"])
        sim1 = float(row["similarity_1"])
        sim2 = float(row["similarity_2"])
        sim3 = float(row["similarity_3"])

        captions = caption_map.get(candidate, [])

        if captions:
            caption_text = "<br>".join(
                html.escape(str(c))
                for c in captions
            )
        else:
            caption_text = "Caption not found"

        sections.append(
            f"""
            <section class="query">

                <div class="query-header">
                    <h2>{html.escape(query_id)}</h2>

                    <div>
                        Category:
                        <strong>{html.escape(category)}</strong>
                    </div>
                </div>

                <div class="caption">
                    <strong>Modification caption:</strong><br>
                    {caption_text}
                </div>

                <div class="grid">

                    {card(
                        "REFERENCE / CANDIDATE",
                        candidate,
                        category
                    )}

                    {card(
                        "TRUE TARGET",
                        target,
                        category,
                        sim_pos
                    )}

                    {card(
                        "HARD NEGATIVE 1",
                        neg1,
                        neg1_cat,
                        sim1
                    )}

                    {card(
                        "HARD NEGATIVE 2",
                        neg2,
                        neg2_cat,
                        sim2
                    )}

                    {card(
                        "HARD NEGATIVE 3",
                        neg3,
                        neg3_cat,
                        sim3
                    )}

                </div>

            </section>
            """
        )

    return "\n".join(sections)


# ============================================================
# MAIN
# ============================================================

def main():

    all_sections = []

    for category in ["dress", "shirt", "toptee"]:

        print(f"Building visual inspection for: {category}")

        section = build_category_html(category)

        all_sections.append(
            f"""
            <div class="category-section">

                <h1>{category.upper()}</h1>

                {section}

            </div>
            """
        )

    output_file = (
        OUTPUT_ROOT / "phase6_hard_negative_visual_inspection.html"
    )

    html_content = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Phase 6 Hard Negative Visual Inspection</title>

<style>

body {{
    font-family: Arial, sans-serif;
    background: #f4f4f4;
    margin: 20px;
}}

h1 {{
    margin-top: 40px;
    padding: 15px;
    background: #222;
    color: white;
}}

.query {{
    background: white;
    margin: 30px 0;
    padding: 20px;
    border-radius: 10px;
}}

.query-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.caption {{
    background: #f1f1f1;
    padding: 12px;
    margin: 15px 0;
    border-radius: 6px;
    line-height: 1.5;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(5, minmax(180px, 1fr));

    gap: 15px;
}}

.card {{
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 8px;
    background: #fafafa;
}}

.card h3 {{
    font-size: 14px;
    min-height: 35px;
}}

.card img {{
    width: 100%;
    height: 260px;
    object-fit: contain;
    background: white;
    border: 1px solid #ddd;
}}

.asin {{
    font-family: monospace;
    margin-top: 8px;
    font-size: 12px;
}}

.category {{
    font-size: 12px;
    margin-top: 4px;
}}

.similarity {{
    font-weight: bold;
    margin-top: 6px;
}}

.missing {{
    height: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #ddd;
    color: #900;
    font-weight: bold;
}}

.category-section {{
    margin-bottom: 50px;
}}

</style>

</head>

<body>

<h1>
Phase 6 — Hard Negative Visual Inspection
</h1>

<p>
Inspect whether each hard negative is genuinely visually similar
to the reference/target and whether it differs in the relevant
fine-grained attributes.
</p>

<p>
<strong>Expected:</strong>
hard negatives should be visually confusing but should not be
the actual target or an exact duplicate.
</p>

{''.join(all_sections)}

</body>

</html>
"""

    output_file.write_text(
        html_content,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("VISUAL INSPECTION GALLERY CREATED")
    print("=" * 60)
    print(output_file)

    webbrowser.open(output_file.resolve().as_uri())


if __name__ == "__main__":
    main()