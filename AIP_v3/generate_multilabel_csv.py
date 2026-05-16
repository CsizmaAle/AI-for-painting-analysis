import csv
import ast
import os
import random
from label_mapping import GENRE_TO_TAGS, GENRE_TO_PERIOD, ALL_TAGS, SUBJECT_TAGS

INPUT_CSV   = "archive/classes.csv"
OUTPUT_CSV  = "multilabel_dataset.csv"
CLIP_CSV    = "clip_subjects.csv"
ARCHIVE_DIR = "archive"
RANDOM_SEED = 42
TRAIN_RATIO = 0.8

SUBJECT_TAG_SET = set(SUBJECT_TAGS)


def parse_genre(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g.strip().replace(" ", "_") for g in genres]
    except Exception:
        return [genre_str.strip().replace(" ", "_")]


def year_to_period(year_str):
    """Maps a year string ('1890', 'c. 1890', '1880-1895') to a PERIOD_TAG."""
    try:
        for token in str(year_str).replace("-", " ").replace("c.", "").split():
            if token.isdigit() and len(token) == 4:
                year = int(token)
                if year < 1400: return "pre_renaissance"
                if year < 1600: return "renaissance_era"
                if year < 1700: return "baroque_era"
                if year < 1800: return "18th_century"
                if year < 1900: return "19th_century"
                if year < 1960: return "early_modern"
                return "contemporary"
    except Exception:
        pass
    return None


def load_clip_subjects(clip_csv):
    """Returns dict mapping filename -> set of subject tags from CLIP."""
    if not os.path.exists(clip_csv):
        return {}
    clip_data = {}
    with open(clip_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            clip_data[fn] = {tag for tag in SUBJECT_TAGS if row.get(tag, "0") == "1"}
    return clip_data


def generate(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    clip_subjects = load_clip_subjects(CLIP_CSV)
    if clip_subjects:
        print(f"CLIP subject labels loaded: {len(clip_subjects):,} images")
    else:
        print("clip_subjects.csv not found — falling back to genre mapping for subjects")

    skipped = 0
    rows = []

    with open(input_csv, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []

        split_col = next((c for c in ("split", "subset") if c in fieldnames), None)
        has_year  = "year" in fieldnames

        for row in reader:
            filename = row["filename"]
            genres   = parse_genre(row["genre"])

            tags       = set()
            period_tag = None

            for genre in genres:
                # only take non-subject tags from genre mapping
                tags.update(t for t in GENRE_TO_TAGS.get(genre, []) if t not in SUBJECT_TAG_SET)
                if period_tag is None:
                    period_tag = GENRE_TO_PERIOD.get(genre)

            # year-based period overrides genre-based when available
            if has_year:
                yp = year_to_period(row.get("year", ""))
                if yp:
                    period_tag = yp

            if period_tag:
                tags.add(period_tag)

            # subject tags: CLIP when available, genre mapping as fallback
            if filename in clip_subjects:
                tags.update(clip_subjects[filename])
            else:
                for genre in genres:
                    tags.update(t for t in GENRE_TO_TAGS.get(genre, []) if t in SUBJECT_TAG_SET)

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
        cut = int(len(rows) * TRAIN_RATIO)
        rows = [(fn, "train", tv) for fn, _, tv in rows[:cut]] + \
               [(fn, "test",  tv) for fn, _, tv in rows[cut:]]

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["filename", "subset"] + ALL_TAGS)
        for filename, subset, tag_vector in rows:
            writer.writerow([filename, subset] + tag_vector)

    train_count = sum(1 for _, s, _ in rows if s == "train")
    test_count  = sum(1 for _, s, _ in rows if s == "test")
    print(f"Done. Train: {train_count:,}  |  Test: {test_count:,}  |  Skipped: {skipped:,}")
    print(f"Tags per row: {len(ALL_TAGS)}  |  Output: {output_csv}")


if __name__ == "__main__":
    generate()
