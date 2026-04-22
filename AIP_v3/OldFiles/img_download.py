import os
import requests
import csv
from urllib.parse import quote

k=open("metadata1.csv","w+")

writer=csv.writer(k)
header="img_path, plaja, campie, deal, munte, lac, rau, padure, mare, oras, sat"
writer.writerow(header)
    

# Setează cheia API obținută de la Pixabay
API_KEY = '47733631-54e282f003ab41627417302d4'

# Funcție pentru a descărca imagini de pe Pixabay
def download_images_from_pixabay(query, num_images, output_folder):
    # URL-ul de bază pentru API-ul Pixabay
    url = "https://pixabay.com/api/"
    
    # Parametrii pentru cererea API
    params = {
        'key': API_KEY,
        'q': query,
        'image_type': 'photo',  # Poți schimba tipul de imagine
        'per_page': num_images,  # Limita de imagini
        'safe_search': 'true',   # Căutare sigură
        'orientation': 'horizontal'  # Opțional: doar imagini orizontale
    }
    
    # Trimite cererea GET
    response = requests.get(url, params=params)
    
    # Verifică dacă cererea a avut succes
    if response.status_code == 200:
        # Parseați rezultatele JSON
        data = response.json()
        
        # Creează folderul de ieșire dacă nu există
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Descărcați fiecare imagine
        for i, hit in enumerate(data['hits']):
            # URL-ul imaginii
            image_url = hit['webformatURL']
            
            # Descărcați imaginea
            img_data = requests.get(image_url).content
            
            # Numele fișierului
            img_name = os.path.join(output_folder, f"{query}_{i + 1}.jpg")
            
            # Scrieți fișierul
            with open(img_name, 'wb') as f:
                f.write(img_data)
            
            label = find_label(query)
            text= "photos4/" + img_name + label
            
            writer.writerow(text)
            
            print(f"Imaginea {i + 1} a fost salvată ca {img_name}")
    else:
        print("Eroare la cererea API: ", response.status_code)

# Exemplu de utilizare
query = "mountains"  # Cuvântul cheie pentru căutare
num_images = 6  # Numărul de imagini dorite
output_folder = "photos2"  # Folderul de destinație pentru imagini
download_images_from_pixabay(query, num_images, output_folder)

def find_label(word):
    possible_labels=["beach","", "", "mountain", "lake", "river", "forest", "sea", "city", "village"]
    #plaja, campie, deal, munte, lac, rau, padure, mare, oras, sat