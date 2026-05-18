# tag categories: period, style, mood, subject, visual elements
import csv
import os
import random
from labels_prompts import label_mapping_v4 as lm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV =os.path.join(BASE_DIR, "../../archive/classes.csv")
OUTPUT_CSV =os.path.join(BASE_DIR, "../data/multilabel_dataset.csv")
IMAGE_LABELS_CLIP =os.path.join(BASE_DIR, "../data/clip_subjects.csv")



def load_csv(csv_file):
    data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data.append(row)
    return data

def get_style_tags(filename):
    genre_folder = filename.split("/")[0]  # "Impressionism/monet_foo.jpg" â†’ "Impressionism"
    styles = lm.GENRE_TO_STYLE.get(genre_folder, [])
    return [1 if tag in styles else 0 for tag in lm.STYLE_TAGS]


def get_period_vector(filename):
    genre_folder = filename.split("/")[0]
    period = lm.GENRE_TO_PERIOD.get(genre_folder)
    return [1 if tag == period else 0 for tag in lm.PERIOD_TAGS]

def get_subset(subset_str):
    if subset_str== "train":
        return "train"
    else: 
        return "test" if random.randint(0, 1) == 0 else "validation"

def main():
    classes_data = load_csv(INPUT_CSV)
    clip_labels_data = load_csv(IMAGE_LABELS_CLIP)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + ["subset"] + lm.PERIOD_TAGS+ lm.STYLE_TAGS + list(clip_labels_data[0].keys())[1:]   # style is fallback for missing clip labels
        writer.writerow(header)
        
        random.seed(42)
        
        clip_dict = {r["filename"]: r for r in clip_labels_data}
        for row in classes_data:
            filename = row["filename"]
            subset=get_subset(row["subset"])

            style_tags = get_style_tags(filename)
            period_vec = get_period_vector(filename)
            
            clip_row = clip_dict.get(filename)
            if clip_row:
                clip_tags = [int(clip_row[tag]) for tag in header[len(lm.PERIOD_TAGS) + len(lm.STYLE_TAGS) + 2:]]
            else:
                clip_tags = [0] * (len(header) - len(lm.PERIOD_TAGS) - len(lm.STYLE_TAGS) - 2)
            
            
            
            output_row = [filename]+ [subset] + period_vec + style_tags + clip_tags
            writer.writerow(output_row)

if __name__ == "__main__":
    main()