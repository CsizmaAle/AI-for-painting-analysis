#descarcare automata a imaginilor, dar nu merge asa ca trb facuta manual -  scopuri educative
# se vor lua manual pozele si se vor pune in photos, iar de acolo se vor introduce sub forma: titlu.jpg, categorie
import requests 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from bs4 import BeautifulSoup
from PIL import Image
import os
import sys
import numpy as np
import random
import csv
import shutil


#resize and normalize the image folder
def resize_normalize_images(input_folder, output_folder, size=(300, 300)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
                    
            try:
               with Image.open(input_path) as img:
                   
                   img=img.resize(size)
                   
                   img=img.convert("RGB")
                   img_array=np.array(img, dtype=np.float32)
                   img_normalized=img_array/255.0
                   img_normalized=(img_normalized*255).astype(np.uint8)
                   
                   Image.fromarray(img_normalized).save(output_path)
                   print(f"Normalized and saved: {output_path}")
                    
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                    
def split_dataset(input_folder, output_folder, train_size=0.8):
    
    # Creează directoarele pentru train și test
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    clear_folder("dataset_split/train")
    clear_folder("dataset_split/test")

    # Listează toate imaginile din folderul de intrare
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # Amestecă imaginile

    # Împarte imaginile în train și test
    num_train = int(len(images) * train_size)
    train_images = images[:num_train]
    test_images = images[num_train:]

    # Copiază imaginile în directoarele corespunzătoare
    for img in train_images:
        shutil.copy(os.path.join(input_folder, img), os.path.join(train_folder, img))
    for img in test_images:
        shutil.copy(os.path.join(input_folder, img), os.path.join(test_folder, img))

    print(f"Imaginile au fost împărțite: {len(train_images)} pentru antrenament, {len(test_images)} pentru test.")

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Șterge fișierul sau link-ul
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Șterge subfolderele
        except Exception as e:
            print(f"Eroare la ștergerea {file_path}: {e}")         
            
def create_csv_split_dataset(input_folder, csv_input_file, csv_output_file):
    out_file= open(csv_output_file, "w+", newline='')
    in_file=open(csv_input_file)
    
    reader = csv.reader(in_file)
    writer = csv.writer(out_file)
    
    header=next(reader)
    writer.writerow(header)
    
    for line in reader:
        
        try:    
            img_name=extract_img_name(line[0])
            
            if img_name in os.listdir(input_folder):
            
                img_name= input_folder+"/"+img_name
        
                new_line=[]
                new_line.append(img_name)
                for i in range(1, len(line)):
                    new_line.append(line[i])
        
                writer.writerow(new_line)
                    
        except Exception as e:
            print(f"Failed to process {line}: {e}") 
          
def extract_img_name(text):
    name=""
    ok=0
    for x in text:
        if ok==1:
            name+=x
        if x=="/":
            ok=1
        
    return name

#numpy arrays and encode labels
def create_vectors(csv_input_file):
    input_file=open(csv_input_file, "r")
    
    reader=csv.reader(input_file)
    header=next(reader)
    
    x=[]
    y=[]
    
    for line in reader:
        try:
            img_path=line[0]
            labels=line[1:]
            img = Image.open(img_path).convert('RGB') 
            img_array = np.array(img)
            x.append(img_array.flatten())  # Aplatizează imaginea într-un vector
            y.append(labels)
            
        except Exception as e:
            print(f"Failed to process {line}: {e}")
        
    x = np.array(x)
    y = np.array(y)
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    
    return x, y
            
    
    