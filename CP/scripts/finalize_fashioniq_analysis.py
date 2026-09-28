import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FASHIONIQ = ROOT / "data" / "raw" / "fashioniq"
PROCESSED = ROOT / "data" / "processed" / "fashioniq"

OUTPUT_JSON = PROCESSED / "final_dataset_statistics.json"
OUTPUT_CSV = PROCESSED / "caption_concept_statistics.csv"


CATEGORIES = ["dress", "shirt", "toptee"]
SPLITS = ["train", "val", "test"]


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return normalize_text(text).split()


def get_caption_files():
    files = {}

    for category in CATEGORIES:
        files[category] = {}

        for split in SPLITS:
            path = (
                FASHIONIQ
                / "captions"
                / f"cap.{category}.{split}.json"
            )

            if path.exists():
                files[category][split] = path

    return files


def get_image_dir(category):
    return FASHIONIQ / category


def image_exists(category, image_id):
    image_path = get_image_dir(category) / f"{image_id}.jpg"
    return image_path.exists()


# ============================================================
# BASIC IMAGE STATISTICS
# ============================================================

def analyze_images():
    result = {}

    for category in CATEGORIES:
        image_dir = get_image_dir(category)

        if image_dir.exists():
            images = list(image_dir.glob("*.jpg"))
        else:
            images = []

        result[category] = {
            "local_images": len(images)
        }

    return result


# ============================================================
# CAPTION ANALYSIS
# ============================================================

def analyze_captions():

    statistics = {}

    all_concepts = Counter()
    all_words = Counter()

    for category in CATEGORIES:

        statistics[category] = {}

        for split in SPLITS:

            caption_path = (
                FASHIONIQ
                / "captions"
                / f"cap.{category}.{split}.json"
            )

            if not caption_path.exists():
                continue

            data = load_json(caption_path)

            query_count = len(data)

            caption_count = 0
            caption_lengths = []

            unique_candidates = set()
            unique_targets = set()

            affected_by_missing_image = 0
            missing_reference = 0
            missing_target = 0

            split_concepts = Counter()
            split_words = Counter()

            for item in data:

                candidate = item.get("candidate")
                target = item.get("target")

                if candidate:
                    unique_candidates.add(candidate)

                if target:
                    unique_targets.add(target)

                captions = item.get("captions", [])

                caption_count += len(captions)

                for caption in captions:

                    tokens = tokenize(caption)

                    caption_lengths.append(len(tokens))

                    for word in tokens:
                        split_words[word] += 1
                        all_words[word] += 1

                    # ------------------------------------------------
                    # Lightweight concept extraction.
                    #
                    # These are NOT ground-truth labels.
                    # They only describe modification language.
                    # ------------------------------------------------

                    concept_groups = {
                        "color": [
                            "black", "white", "red", "blue",
                            "green", "yellow", "pink", "purple",
                            "orange", "brown", "grey", "gray",
                            "beige", "navy"
                        ],

                        "pattern": [
                            "floral", "flower", "striped",
                            "stripe", "plaid", "checkered",
                            "checked", "polka", "pattern",
                            "print", "printed", "solid"
                        ],

                        "sleeve": [
                            "sleeve", "sleeves", "sleeveless",
                            "strap", "straps", "short-sleeve",
                            "long-sleeve"
                        ],

                        "length": [
                            "longer", "shorter", "long",
                            "short", "length", "mini",
                            "midi", "maxi"
                        ],

                        "shape_fit": [
                            "loose", "tight", "fitted",
                            "fit", "slim", "wide",
                            "flowy", "straight"
                        ],

                        "neckline": [
                            "neck", "neckline", "v-neck",
                            "crew", "collar", "halter"
                        ],

                        "material": [
                            "cotton", "silk", "denim",
                            "leather", "wool", "polyester",
                            "linen", "fabric"
                        ],

                        "style": [
                            "casual", "formal", "sporty",
                            "elegant", "classic", "simple",
                            "plain"
                        ]
                    }

                    normalized = normalize_text(caption)

                    for concept, vocabulary in concept_groups.items():

                        if any(
                            re.search(
                                rf"\b{re.escape(term)}\b",
                                normalized
                            )
                            for term in vocabulary
                        ):
                            split_concepts[concept] += 1
                            all_concepts[concept] += 1

                # ----------------------------------------------------
                # Check local image availability
                # ----------------------------------------------------

                candidate_available = (
                    candidate is not None
                    and image_exists(category, candidate)
                )

                # Test annotations intentionally have no target.
                target_available = (
                    target is None
                    or image_exists(category, target)
                )

                if not candidate_available:
                    missing_reference += 1

                if target is not None and not target_available:
                    missing_target += 1

                if not candidate_available or not target_available:
                    affected_by_missing_image += 1

            statistics[category][split] = {

                "queries": query_count,

                "captions": caption_count,

                "average_captions_per_query": (
                    caption_count / query_count
                    if query_count
                    else 0
                ),

                "average_caption_words": (
                    sum(caption_lengths) / len(caption_lengths)
                    if caption_lengths
                    else 0
                ),

                "unique_candidates": len(unique_candidates),

                "unique_targets": (
                    len(unique_targets)
                    if split != "test"
                    else None
                ),

                "missing_reference_queries": missing_reference,

                "missing_target_queries": (
                    missing_target
                    if split != "test"
                    else None
                ),

                "affected_queries": affected_by_missing_image,

                "modification_concepts": dict(
                    split_concepts.most_common()
                ),

                "top_words": dict(
                    split_words.most_common(30)
                )
            }

    return statistics, all_concepts, all_words


# ============================================================
# CLEAN QUERY STATISTICS
# ============================================================

def analyze_clean_queries():

    statistics = {}

    total_original = 0
    total_usable = 0
    total_excluded = 0

    excluded_reasons = Counter()

    for category in CATEGORIES:

        statistics[category] = {}

        for split in SPLITS:

            path = (
                PROCESSED
                / f"{split}_queries_{category}.csv"
            )

            if not path.exists():
                continue

            df = pd.read_csv(path)

            usable = len(df)

            # Reconstruct original query count from raw captions.
            raw_path = (
                FASHIONIQ
                / "captions"
                / f"cap.{category}.{split}.json"
            )

            raw_data = load_json(raw_path)

            original = len(raw_data)

            excluded = original - usable

            total_original += original
            total_usable += usable
            total_excluded += excluded

            statistics[category][split] = {
                "original_queries": original,
                "usable_queries": usable,
                "excluded_queries": excluded,
                "usable_percentage": (
                    usable / original * 100
                    if original
                    else 0
                )
            }

    excluded_path = PROCESSED / "excluded_queries.csv"

    if excluded_path.exists():

        excluded_df = pd.read_csv(excluded_path)

        for reason in excluded_df["reason"].dropna():
            excluded_reasons[reason] += 1

    return (
        statistics,
        {
            "original_queries": total_original,
            "usable_queries": total_usable,
            "excluded_queries": total_excluded,
            "usable_percentage": (
                total_usable / total_original * 100
                if total_original
                else 0
            ),
            "excluded_reasons": dict(excluded_reasons)
        }
    )


# ============================================================
# SAVE CONCEPT CSV
# ============================================================

def save_concept_csv(all_concepts, all_words):

    rows = []

    total_concept_mentions = sum(all_concepts.values())

    for concept, count in all_concepts.most_common():

        rows.append({
            "concept": concept,
            "caption_count": count,
            "percentage_of_concept_mentions": (
                count / total_concept_mentions * 100
                if total_concept_mentions
                else 0
            )
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT_CSV,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL FASHIONIQ PHASE-1 DATASET ANALYSIS")
    print("=" * 70)

    image_stats = analyze_images()

    caption_stats, all_concepts, all_words = analyze_captions()

    clean_stats, clean_totals = analyze_clean_queries()

    report = {

        "project": "Attribute-Aware Cross-Domain Composed Image Retrieval",

        "dataset": "FashionIQ",

        "phase": "Phase 1 - Dataset Preparation and Analysis",

        "image_statistics": image_stats,

        "query_statistics": clean_stats,

        "overall_clean_dataset": clean_totals,

        "caption_statistics": caption_stats,

        "modification_language_summary": {
            "concept_counts": dict(
                all_concepts.most_common()
            ),
            "top_words": dict(
                all_words.most_common(100)
            )
        },

        "attribute_annotation_note": (
            "FashionIQ does not contain a separate categorical "
            "attribute annotation file in the downloaded dataset. "
            "Concept counts in this report are derived from "
            "natural-language modification captions and are NOT "
            "treated as ground-truth categorical attribute labels."
        ),

        "test_split_note": (
            "FashionIQ test caption files do not contain target IDs. "
            "Therefore target availability is not evaluated for test "
            "queries."
        ),

        "missing_image_policy": (
            "Queries whose required local images are unavailable "
            "are excluded from the processed query files. "
            "Original raw annotations remain unchanged."
        )
    }

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    save_concept_csv(
        all_concepts,
        all_words
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print("\nIMAGE STATISTICS")
    print("-" * 70)

    for category, stats in image_stats.items():

        print(
            f"{category:8s} | "
            f"local images = {stats['local_images']}"
        )

    print("\nCLEAN QUERY STATISTICS")
    print("-" * 70)

    for category, splits in clean_stats.items():

        for split, stats in splits.items():

            print(
                f"{category:8s} | "
                f"{split:5s} | "
                f"original = {stats['original_queries']:5d} | "
                f"usable = {stats['usable_queries']:5d} | "
                f"excluded = {stats['excluded_queries']:4d} | "
                f"usable = {stats['usable_percentage']:.2f}%"
            )

    print("\nOVERALL")
    print("-" * 70)

    print(
        f"Original queries : "
        f"{clean_totals['original_queries']}"
    )

    print(
        f"Usable queries   : "
        f"{clean_totals['usable_queries']}"
    )

    print(
        f"Excluded queries : "
        f"{clean_totals['excluded_queries']}"
    )

    print(
        f"Usable percentage: "
        f"{clean_totals['usable_percentage']:.2f}%"
    )

    print("\nMODIFICATION CONCEPTS")
    print("-" * 70)

    for concept, count in all_concepts.most_common():

        print(
            f"{concept:12s} | "
            f"{count:6d} caption mentions"
        )

    print("\nOUTPUT FILES")
    print("-" * 70)

    print(OUTPUT_JSON)
    print(OUTPUT_CSV)

    print("\nPhase 1 analysis completed successfully.")


if __name__ == "__main__":
    main()