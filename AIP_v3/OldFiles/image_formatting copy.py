#descarcare automata a imaginilor, dar nu merge asa ca trb facuta manual -  scopuri educative
# se vor lua manual pozele si se vor pune in photos, iar de acolo se vor introduce sub forma: titlu.jpg, categorie
import requests 
from bs4 import BeautifulSoup
from PIL import Image
import os
import sys
import numpy as np
import random
import shutil


#resize and normalize the image folder
def resize_images(input_folder, output_folder, size=(300, 300)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Deschide imaginea
                with Image.open(input_path) as img:
                    # Redimensionează imaginea
                    img_resized = img.resize(size)
                    
                    # Salvează imaginea redimensionată
                    img_resized.save(output_path)
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


resize_images(
    input_folder="photos", 
    output_folder="photos2", 
    size=(300, 300)  # Redimensionează la 800x800
)



#normalizare imagine
def normalize_image(image_path):
    # Deschide imaginea și convertește-o în format RGB
    img = Image.open(image_path).convert('RGB')

    # Convertește imaginea într-un array NumPy
    img_array = np.array(img, dtype=np.float32)

    # Normalizează valorile la intervalul [0, 1]
    img_normalized = img_array / 255.0

    return img_normalized

#normalized_image = normalize_image("path_to_image.jpg")
#print(normalized_image)

def normalize_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Normalizează imaginea
            normalized_img = normalize_image(input_path)

            # Convertim matricea normalizată înapoi într-o imagine
            normalized_img = (normalized_img * 255).astype(np.uint8)
            Image.fromarray(normalized_img).save(output_path)
            print(f"Normalized and saved: {output_path}")

# Exemplu de utilizare
normalize_images_in_folder("photos2", "photos3")




def split_dataset(input_folder, output_folder):
    """
    Împarte imaginile într-un set de antrenament și unul de test.
    :param input_folder: Folderul cu imaginile originale.
    :param output_folder: Folderul unde vor fi salvate seturile de antrenament și test.
    :param train_size: Proporția de date pentru antrenament (ex. 80%).
    """
    os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)

    for label in os.listdir(input_folder):
        label_path = os.path.join(input_folder, label)
        if os.path.isdir(label_path):  # Dacă este un subfolder cu imagini
            images = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)  # Amestecă listele de imagini
            
            # Împărțirea în train și test
            num_train = int(len(images) * 0.8)
            train_images = images[:num_train]
            test_images = images[num_train:]

            # Copiază imaginile în folderele corespunzătoare
            
            for img in train_images:
                shutil.copy(os.path.join(label_path, img), os.path.join(output_folder, 'train', label, img))
            for img in test_images:
                shutil.copy(os.path.join(label_path, img), os.path.join(output_folder, 'test', label, img))
            
            
                

# Exemplu de utilizare
split_dataset("photos3", "dataset_split")