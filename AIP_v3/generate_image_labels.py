"""
Generates per-image labels for subject, mood, and element tags.
  Subject + Mood  : CLIP ViT-B/32 cosine similarity (semantic)
  Element         : pixel analysis (objectively measurable)
                    geometric_shapes is excluded — stays in genre mapping

Output: image_labels.csv  (one row per image)
Resumable: already-processed images are skipped on restart.
"""

import os
import csv
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

from label_mapping import SUBJECT_TAGS, MOOD_TAGS

ARCHIVE_DIR     = "../archive"
CLASSES_CSV     = "../archive/classes.csv"
OUTPUT_CSV      = "image_labels.csv"
BATCH_SIZE      = 64

SUBJECT_THRESH  = 0.24
MOOD_MIN_THRESH = 0.19   # moods are subtler than concrete subjects
MOOD_TOP_K      = 2      # at most 2 moods assigned per image

# Element tags handled by pixel analysis (geometric_shapes excluded)
PIXEL_ELEMENTS = [
    "warm_tones", "cool_tones", "vivid_colors", "muted_tones",
    "high_contrast", "soft_light", "dark_shadows", "rich_texture",
]

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

MOOD_PROMPTS = {
    "joyful": [
        "a joyful and cheerful painting radiating happiness",
        "a bright optimistic painting full of joy and warmth",
    ],
    "melancholic": [
        "a melancholic and sorrowful painting",
        "a sad wistful painting with a sense of loss or longing",
    ],
    "dramatic": [
        "a dramatic and intense painting with powerful tension",
        "a bold dramatic scene with strong visual impact",
    ],
    "peaceful": [
        "a peaceful and calm painting",
        "a tranquil painting with a restful gentle atmosphere",
    ],
    "dark": [
        "a dark gloomy painting with an oppressive atmosphere",
        "a painting with a dark foreboding mood",
    ],
    "mysterious": [
        "a mysterious and enigmatic painting",
        "a painting with an air of mystery and the unknown",
    ],
    "tense": [
        "a tense and anxious painting conveying unease",
        "a painting with psychological tension and disquiet",
    ],
    "spiritual": [
        "a spiritual and devotional painting with sacred imagery",
        "a painting conveying divine transcendence and spirituality",
    ],
    "energetic": [
        "an energetic dynamic painting full of movement and vitality",
        "a vibrant active painting with strong kinetic energy",
    ],
    "serene": [
        "a serene harmonious painting with gentle quiet beauty",
        "a calm meditative painting with peaceful elegance",
    ],
}


def build_text_matrix(model, device):
    """Pre-compute averaged text embeddings: subjects first, then moods."""
    tags_in_order = SUBJECT_TAGS + MOOD_TAGS
    all_prompts   = {**SUBJECT_PROMPTS, **MOOD_PROMPTS}
    embeddings = []
    with torch.no_grad():
        for tag in tags_in_order:
            tokens = clip.tokenize(all_prompts[tag]).to(device)
            embs   = model.encode_text(tokens)
            embs   = embs / embs.norm(dim=-1, keepdim=True)
            embeddings.append(embs.mean(dim=0))
    mat = torch.stack(embeddings)
    return mat / mat.norm(dim=-1, keepdim=True)


def assign_subjects(sim_row):
    n     = len(SUBJECT_TAGS)
    sims  = sim_row[:n]
    labels = [1 if s >= SUBJECT_THRESH else 0 for s in sims]
    if sum(labels) == 0:
        labels[int(np.argmax(sims))] = 1  # guarantee at least one
    return labels


def assign_moods(sim_row):
    sims    = sim_row[len(SUBJECT_TAGS):]
    indexed = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    selected = set()
    for idx, score in indexed:
        if score >= MOOD_MIN_THRESH and len(selected) < MOOD_TOP_K:
            selected.add(idx)
    if not selected:
        selected.add(indexed[0][0])  # guarantee at least one
    return [1 if i in selected else 0 for i in range(len(MOOD_TAGS))]


def analyze_pixels(pil_img):
    """Returns list of 0/1 for PIXEL_ELEMENTS derived from raw image data."""
    img = np.array(pil_img.resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # HSV saturation
    sat = np.where(max_c > 0.01, delta / (max_c + 1e-8), 0.0)

    # Hue (0–1 scale)
    hue = np.zeros_like(max_c)
    mask = delta > 0.01
    rm = mask & (max_c == r)
    gm = mask & (max_c == g)
    bm = mask & (max_c == b)
    hue[rm] = ((g[rm] - b[rm]) / (delta[rm] + 1e-8)) % 6
    hue[gm] = 2.0 + (b[gm] - r[gm]) / (delta[gm] + 1e-8)
    hue[bm] = 4.0 + (r[bm] - g[bm]) / (delta[bm] + 1e-8)
    hue = hue / 6.0

    h, w = hue.flatten(), sat.flatten()
    w_sum = w.sum() + 1e-8

    # Warm: red/orange/yellow (0–60° and 330–360°)
    warm_w = w * ((h < 0.167) | (h > 0.917)).astype(np.float32)
    # Cool: blue/cyan (150–270°)
    cool_w = w * ((h > 0.417) & (h < 0.75)).astype(np.float32)
    warm_ratio = warm_w.sum() / w_sum
    cool_ratio = cool_w.sum() / w_sum

    lum       = 0.299 * r + 0.587 * g + 0.114 * b
    lum_std   = lum.std()
    lum_mean  = lum.mean()
    dark_ratio = (lum < 0.2).mean()
    mean_sat  = sat.mean()

    pad = np.pad(lum, 1, mode="edge")
    lap = -pad[:-2, 1:-1] - pad[2:, 1:-1] - pad[1:-1, :-2] - pad[1:-1, 2:] + 4 * lum
    texture = lap.var()

    out = {}

    # warm_tones XOR cool_tones
    if warm_ratio > cool_ratio and warm_ratio > 0.15:
        out["warm_tones"] = 1; out["cool_tones"] = 0
    elif cool_ratio > warm_ratio and cool_ratio > 0.15:
        out["cool_tones"] = 1; out["warm_tones"] = 0
    else:
        out["warm_tones"] = 0; out["cool_tones"] = 0

    # vivid_colors XOR muted_tones
    out["vivid_colors"] = 1 if mean_sat > 0.28 else 0
    out["muted_tones"]  = 1 - out["vivid_colors"]

    # high_contrast XOR soft_light (both 0 if ambiguous)
    if lum_std > 0.18:
        out["high_contrast"] = 1; out["soft_light"] = 0
    elif lum_std < 0.12 and lum_mean > 0.40:
        out["soft_light"] = 1; out["high_contrast"] = 0
    else:
        out["high_contrast"] = 0; out["soft_light"] = 0

    out["dark_shadows"] = 1 if dark_ratio > 0.20 else 0
    out["rich_texture"]  = 1 if texture   > 0.004 else 0

    return [out[tag] for tag in PIXEL_ELEMENTS]


def load_done(output_csv):
    if not os.path.exists(output_csv):
        return set()
    with open(output_csv, "r", encoding="utf-8") as f:
        return {row["filename"] for row in csv.DictReader(f)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading CLIP ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    print("Computing text embeddings (subjects + moods)...")
    text_matrix = build_text_matrix(model, device)

    all_files = []
    with open(CLASSES_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            if os.path.exists(os.path.join(ARCHIVE_DIR, fn)):
                all_files.append(fn)
    print(f"Total images on disk: {len(all_files):,}")

    done       = load_done(OUTPUT_CSV)
    to_process = [fn for fn in all_files if fn not in done]
    print(f"Already done: {len(done):,}  |  Remaining: {len(to_process):,}")

    header       = ["filename"] + SUBJECT_TAGS + MOOD_TAGS + PIXEL_ELEMENTS
    write_header = len(done) == 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(header)

        for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Labelling images"):
            batch_files  = to_process[i : i + BATCH_SIZE]
            raw_imgs, clip_tensors, valid = [], [], []

            for fn in batch_files:
                try:
                    pil = Image.open(os.path.join(ARCHIVE_DIR, fn)).convert("RGB")
                    raw_imgs.append(pil)
                    clip_tensors.append(preprocess(pil))
                    valid.append(fn)
                except Exception:
                    pass

            if not valid:
                continue

            img_tensor = torch.stack(clip_tensors).to(device)
            with torch.no_grad():
                img_emb = model.encode_image(img_tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            sims = (img_emb @ text_matrix.T).cpu().numpy()

            for idx, (fn, pil) in enumerate(zip(valid, raw_imgs)):
                sim_row  = sims[idx]
                subjects = assign_subjects(sim_row)
                moods    = assign_moods(sim_row)
                elements = analyze_pixels(pil)
                writer.writerow([fn] + subjects + moods + elements)

        f_out.flush()

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
