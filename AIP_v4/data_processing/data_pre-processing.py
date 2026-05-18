import torch
import clip
from PIL import Image
from data import prompts as pr
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "../../archive")
CLASSES_CSV = os.path.join(ARCHIVE_DIR, "classes.csv")
OUTPUT_CSV  = os.path.join(BASE_DIR, "../data/clip_subjects.csv")
BATCH_SIZE  = 64
THRESHOLD   = 0.24

SUBJECT_TAGS = list(pr.SUBJECT_PROMPTS.keys())
MOOD_TAGS = list(pr.MOOD_PROMPTS.keys())
ELEMENT_TAGS = list(pr.ELEMENT_PROMPTS.keys())

CATEGORIES = {
    "subject":  (pr.SUBJECT_PROMPTS,  0.24, True),   
    "mood":     (pr.MOOD_PROMPTS,     0.20, True),
    "elements": (pr.ELEMENT_PROMPTS,  0.22, False),
}


def compute_text_embeddings(model, device, category):
    prompts, threshold, multi_label = CATEGORIES[category]
    text_embeddings = []
    print(f"Computing text embeddings for {category}...")
    with torch.no_grad():
        for tag in prompts:
            tokens = clip.tokenize(prompts[tag]).to(device)
            embs   = model.encode_text(tokens)
            embs   = embs / embs.norm(dim=-1, keepdim=True)
            text_embeddings.append(embs.mean(dim=0))
    text_matrix = torch.stack(text_embeddings)                      # (num_tags, dim)
    text_matrix = text_matrix / text_matrix.norm(dim=-1, keepdim=True)
    return text_matrix, threshold, multi_label


def generate_csv(cvs_output_file, data):
    with open(cvs_output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename"] + SUBJECT_TAGS + MOOD_TAGS + ELEMENT_TAGS
        writer.writerow(header)
        for row in data:
            writer.writerow(row)


def gather_image_files(cvs_input_file, archive_dir):
    all_files = []
    with open(cvs_input_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            if os.path.exists(os.path.join(archive_dir, fn)):
                all_files.append(fn)
    print(f"Total images found: {len(all_files):,}")
    return all_files


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    all_files = gather_image_files(CLASSES_CSV, ARCHIVE_DIR)

    # Pre-compute text embeddings for every category
    cat_embeddings = {
        cat: compute_text_embeddings(model, device, cat)
        for cat in CATEGORIES
    }

    all_rows = []
    for batch_start in range(0, len(all_files), BATCH_SIZE):
        batch = all_files[batch_start: batch_start + BATCH_SIZE]

        # Load and preprocess images
        images = []
        valid = []
        for fn in batch:
            try:
                img = preprocess(Image.open(os.path.join(ARCHIVE_DIR, fn)).convert("RGB"))
                images.append(img)
                valid.append(fn)
            except Exception as e:
                print(f"Skipping {fn}: {e}")

        if not images:
            continue

        img_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            img_embs = model.encode_image(img_tensor)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)  # (B, dim)

        # Score each category
        cat_labels = {}
        for cat, (text_matrix, threshold, multi_label) in cat_embeddings.items():
            sims = (img_embs @ text_matrix.T)  # (B, num_tags)
            if multi_label:
                labels = (sims >= threshold).int()
            else:
                best = sims.argmax(dim=-1)
                labels = torch.zeros_like(sims, dtype=torch.int)
                labels[torch.arange(len(best)), best] = 1
            cat_labels[cat] = labels.cpu().tolist()

        for i, fn in enumerate(valid):
            row = [fn]
            row += cat_labels["subject"][i]
            row += cat_labels["mood"][i]
            row += cat_labels["elements"][i]
            all_rows.append(row)

        print(f"Processed {min(batch_start + BATCH_SIZE, len(all_files))}/{len(all_files)}")

    generate_csv(OUTPUT_CSV, all_rows)
    print(f"Done — wrote {len(all_rows)} rows to {OUTPUT_CSV}")
    

if __name__ == "__main__":
    main()
