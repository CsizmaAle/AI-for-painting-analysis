import os
import csv
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

ARCHIVE_DIR = "../archive"
CLASSES_CSV = "../archive/classes.csv"
OUTPUT_CSV  = "clip_subjects.csv"
BATCH_SIZE  = 64
THRESHOLD   = 0.24

# Multiple prompts per subject — averaged to get a more robust embedding
SUBJECT_PROMPTS = {
    "portrait": [
        "a painting of a human portrait or face",
        "a portrait of a person painted on canvas",
    ],
    "landscape": [
        "a landscape painting with nature, fields, mountains, or sky",
        "a painting of an outdoor natural landscape",
    ],
    "still_life": [
        "a still life painting with flowers, fruit, or objects on a table",
        "a painting of inanimate arranged objects",
    ],
    "religious_mythological": [
        "a religious painting with saints or biblical figures",
        "a mythological painting with gods or ancient legends",
    ],
    "historical_battle": [
        "a painting of a military battle or war scene",
        "a historical painting of soldiers fighting",
    ],
    "everyday_life": [
        "a genre painting of ordinary people in daily life",
        "a painting of common people doing everyday activities",
    ],
    "sea_water": [
        "a seascape painting with the ocean or sea",
        "a maritime painting with water, ships, or coastline",
    ],
    "urban_city": [
        "a painting of a city street or urban landscape",
        "an urban scene with buildings and city life",
    ],
    "animals": [
        "a painting featuring animals or wildlife",
        "an animal painting with birds, horses, or other creatures",
    ],
    "nude": [
        "a painting of a nude human figure",
        "a figure painting with an unclothed human body",
    ],
    "abstract_nonrepresentational": [
        "an abstract painting with no recognizable objects or figures",
        "a non-representational artwork with shapes and colors only",
    ],
}

SUBJECT_TAGS = list(SUBJECT_PROMPTS.keys())


def load_done(output_csv):
    if not os.path.exists(output_csv):
        return set()
    done = set()
    with open(output_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["filename"])
    return done


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading CLIP ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Pre-compute one averaged text embedding per subject tag
    print("Computing text embeddings...")
    text_embeddings = []
    with torch.no_grad():
        for tag in SUBJECT_TAGS:
            tokens = clip.tokenize(SUBJECT_PROMPTS[tag]).to(device)
            embs   = model.encode_text(tokens)
            embs   = embs / embs.norm(dim=-1, keepdim=True)
            text_embeddings.append(embs.mean(dim=0))
    text_matrix = torch.stack(text_embeddings)                      # (num_tags, dim)
    text_matrix = text_matrix / text_matrix.norm(dim=-1, keepdim=True)

    # Collect all image filenames from classes.csv
    all_files = []
    with open(CLASSES_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            if os.path.exists(os.path.join(ARCHIVE_DIR, fn)):
                all_files.append(fn)
    print(f"Total images found: {len(all_files):,}")

    done = load_done(OUTPUT_CSV)
    to_process = [fn for fn in all_files if fn not in done]
    print(f"Already processed: {len(done):,}  |  Remaining: {len(to_process):,}")

    write_header = len(done) == 0
    f_out = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow(["filename"] + SUBJECT_TAGS)

    for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="CLIP labelling"):
        batch_files = to_process[i : i + BATCH_SIZE]
        images, valid = [], []

        for fn in batch_files:
            try:
                img = Image.open(os.path.join(ARCHIVE_DIR, fn)).convert("RGB")
                images.append(preprocess(img))
                valid.append(fn)
            except Exception:
                pass  # skip corrupt images

        if not images:
            continue

        img_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        sims = (img_emb @ text_matrix.T).cpu().numpy()  # (batch, num_tags)

        for fn, sim_row in zip(valid, sims):
            labels = [1 if s >= THRESHOLD else 0 for s in sim_row]
            # guarantee at least one subject tag per image
            if sum(labels) == 0:
                labels[int(np.argmax(sim_row))] = 1
            writer.writerow([fn] + labels)

        f_out.flush()

    f_out.close()
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
