import csv
import ast
import os
import random
from label_mapping import (
    GENRE_TO_TAGS, GENRE_TO_PERIOD, ALL_TAGS,
    STYLE_TAGS, TECHNIQUE_TAGS, ELEMENT_TAGS, MOOD_TAGS, SUBJECT_TAGS,
)

INPUT_CSV        = "../archive/classes.csv"
OUTPUT_CSV       = "multilabel_dataset.csv"
IMAGE_LABELS_CSV = "image_labels.csv"
ARCHIVE_DIR      = "../archive"
RANDOM_SEED      = 42
TRAIN_RATIO      = 0.8

# Tags taken from genre mapping (period is handled separately)
GENRE_TAGS = set(STYLE_TAGS) | set(TECHNIQUE_TAGS) | {"geometric_shapes"}

# Tags taken from per-image analysis; genre mapping is fallback only
IMAGE_TAGS = set(SUBJECT_TAGS) | set(MOOD_TAGS) | (set(ELEMENT_TAGS) - {"geometric_shapes"})


def parse_genre(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g.strip().replace(" ", "_") for g in genres]
    except Exception:
        return [genre_str.strip().replace(" ", "_")]


def year_to_period(year_str):
    try:
        for token in str(year_str).replace("-", " ").replace("c.", "").split():
            if token.isdigit() and len(token) == 4:
                year = int(token)
                if year < 1600: return "renaissance_era"
                if year < 1700: return "baroque_era"
                if year < 1800: return "18th_century"
                if year < 1900: return "19th_century"
                if year < 1960: return "early_modern"
                return "contemporary"
    except Exception:
        pass
    return None


def load_image_labels(labels_csv):
    """Returns dict: filename -> {tag: int} for all per-image label columns."""
    if not os.path.exists(labels_csv):
        return {}
    data = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            data[fn] = {k: int(v) for k, v in row.items() if k != "filename"}
    return data


def generate(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    image_labels = load_image_labels(IMAGE_LABELS_CSV)
    if image_labels:
        print(f"Per-image labels loaded: {len(image_labels):,} images")
    else:
        print("image_labels.csv not found — falling back to genre mapping for mood/element/subject")

    skipped = 0
    rows = []

    with open(input_csv, "r", encoding="utf-8") as f_in:
        reader    = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []

        split_col = next((c for c in ("split", "subset") if c in fieldnames), None)
        has_year  = "year" in fieldnames

        for row in reader:
            filename = row["filename"]
            genres   = parse_genre(row["genre"])

            # ── Period ────────────────────────────────────────────────────────
            period_tag = None
            for genre in genres:
                if period_tag is None:
                    period_tag = GENRE_TO_PERIOD.get(genre)
            if has_year:
                yp = year_to_period(row.get("year", ""))
                if yp:
                    period_tag = yp

            # ── Style + technique + geometric_shapes from genre mapping ───────
            genre_derived = set()
            for genre in genres:
                genre_derived.update(
                    t for t in GENRE_TO_TAGS.get(genre, []) if t in GENRE_TAGS
                )

            # ── Mood + element + subject: per-image if available ─────────────
            if filename in image_labels:
                img_tags = {
                    t for t, v in image_labels[filename].items()
                    if v == 1 and t in IMAGE_TAGS
                }
            else:
                img_tags = set()
                for genre in genres:
                    img_tags.update(
                        t for t in GENRE_TO_TAGS.get(genre, []) if t in IMAGE_TAGS
                    )

            tags = genre_derived | img_tags
            if period_tag:
                tags.add(period_tag)

            if not tags:
                skipped += 1
                continue

            if not os.path.exists(os.path.join(ARCHIVE_DIR, filename)):
                skipped += 1
                continue

            subset = row[split_col] if split_col else None
            if subset == "val":
                subset = "test"
            if subset not in ("train", "test"):
                subset = "train"

            tag_vector = [1 if tag in tags else 0 for tag in ALL_TAGS]
            rows.append((filename, subset, tag_vector))

    if not split_col:
        random.seed(RANDOM_SEED)
        random.shuffle(rows)
        cut  = int(len(rows) * TRAIN_RATIO)
        rows = [(fn, "train", tv) for fn, _, tv in rows[:cut]] + \
               [(fn, "test",  tv) for fn, _, tv in rows[cut:]]

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["filename", "subset"] + ALL_TAGS)
        for filename, subset, tag_vector in rows:
            writer.writerow([filename, subset] + tag_vector)

    train_count = sum(1 for _, s, _ in rows if s == "train")
    test_count  = sum(1 for _, s, _ in rows if s == "test")
    src = "per-image labels" if image_labels else "genre mapping (fallback)"
    print(f"Done. Train: {train_count:,}  |  Test: {test_count:,}  |  Skipped: {skipped:,}")
    print(f"Mood / element / subject source: {src}")
    print(f"Tags: {len(ALL_TAGS)}  |  Output: {output_csv}")


if __name__ == "__main__":
    generate()
