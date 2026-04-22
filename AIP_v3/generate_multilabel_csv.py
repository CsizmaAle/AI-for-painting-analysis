import csv
import ast
import os
from label_mapping import GENRE_TO_TAGS, ALL_TAGS

INPUT_CSV = "archive/classes.csv"
OUTPUT_CSV = "multilabel_dataset.csv"
ARCHIVE_DIR = "archive"


def parse_genre(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g.strip() for g in genres]
    except Exception:
        return [genre_str.strip()]


def generate(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    skipped = 0
    written = 0

    with open(input_csv, "r", encoding="utf-8") as f_in, \
         open(output_csv, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)

        writer.writerow(["filename", "subset"] + ALL_TAGS)

        for row in reader:
            filename = row["filename"]
            subset = row.get("subset", "train")
            genres = parse_genre(row["genre"])

            # collect all tags for this image from all its genres
            tags = set()
            for genre in genres:
                if genre in GENRE_TO_TAGS:
                    tags.update(GENRE_TO_TAGS[genre])

            if not tags:
                skipped += 1
                continue

            # verify image file exists
            full_path = os.path.join(ARCHIVE_DIR, filename)
            if not os.path.exists(full_path):
                skipped += 1
                continue

            tag_vector = [1 if tag in tags else 0 for tag in ALL_TAGS]
            writer.writerow([filename, subset] + tag_vector)
            written += 1

    print(f"Done. Written: {written} rows, Skipped: {skipped} rows.")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    generate()
