from PIL import Image
import os
import numpy as np
import random
import csv
import shutil
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


#resize the image folder
def resize_images(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
                    
            try:
               with Image.open(input_path) as img:
                   img=img.resize(size).save(output_path)
                    
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                    

# split the dataset into train and test
def split_dataset(input_folder, output_folder, train_size=0.8):
    
    # Creează directoarele pentru train și test daca nu exista
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    clear_folder(train_folder)
    clear_folder(test_folder)

    # Listeaza toate imaginile din folderul de intrare
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # Amesteca imaginile

    # Imparte imaginile în train si test
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
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
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

# load the images and labels from the csv file
def load_images_and_labels(csv_input_file):
    with open(csv_input_file, "r") as input_file:
        reader=csv.reader(input_file)
        header=next(reader)
        
        x=[]
        y=[]
        
        for line in reader:
            try:
                img_path=line[0]
                labels=line[1:]
                img = Image.open(img_path).convert('RGB').resize((224, 224))
                img_array = np.array(img, dtype=np.float32) 
                img_array = preprocess_input(img_array)  
                x.append(img_array)  
                y.append(labels)
                
            except Exception as e:
                print(f"Failed to process {line}: {e}")
            
        x = np.array(x)
        y = np.array(y)
    
    return x, y
            
    
    