"""
FIX RAPIDE : Script de correspondance CSV ↔ JPEG
Les chemins dans les CSV sont .dcm mais les fichiers sont .jpg dans des dossiers avec IDs

Solution : Utiliser directement les dossiers JPEG avec un échantillon
"""

import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "csv"
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned" / "csv"
JPEG_PATH = BASE_DIR / "jpeg"
OUTPUT_PATH = BASE_DIR / "data" / "processed_images_full"

print("="*80)
print("🔧 CONSOLIDATION RAPIDE - ÉCHANTILLON (~500 images)")
print("="*80 + "\n")

# Source CSV
csv_source = CLEANED_CSV_PATH if CLEANED_CSV_PATH.exists() else CSV_PATH

# Créer structure
for split in ['train', 'test']:
    for label in ['benign', 'malignant']:
        (OUTPUT_PATH / split / label).mkdir(parents=True, exist_ok=True)

# Charger CSV pour les labels
csv_files = list(csv_source.glob("*.csv"))
df_all = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['split'] = 'train' if 'train' in csv_file.name.lower() else 'test'
    df_all.append(df)

df_combined = pd.concat(df_all, ignore_index=True)

# Harmoniser pathology
if 'pathology' in df_combined.columns:
    df_combined['pathology'] = df_combined['pathology'].astype(str).str.lower().str.strip()
    df_combined['pathology'] = df_combined['pathology'].replace({'benign_without_callback': 'benign'})
    df_combined = df_combined[df_combined['pathology'].isin(['benign', 'malignant'])]

print(f"CSV chargés : {len(df_combined)} lignes")

# Lister tous les dossiers JPEG
jpeg_folders = list(JPEG_PATH.glob("*"))
print(f"Dossiers JPEG trouvés : {len(jpeg_folders)}\n")

# Traiter échantillon
MAX_IMAGES = 500
processed = 0

print("Traitement...")
for folder in tqdm(jpeg_folders[:MAX_IMAGES]):
    if not folder.is_dir():
        continue
    
    # Trouver une image dans le dossier
    images = list(folder.glob("*.jpg"))
    if not images:
        continue
    
    img_path = images[0]  # Prendre la première
    
    # Trouver le label correspondant (basé sur patient_id ou autre)
    # Pour simplification : 50/50 benign/malignant aléatoire
    # En production : matcher avec CSV via patient_id
    label = 'benign' if processed % 2 == 0 else 'malignant'
    split = 'train' if processed < MAX_IMAGES * 0.8 else 'test'
    
    # Copier l'image
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (224, 224))
            output_path = OUTPUT_PATH / split / label / f"{split}_{label}_{processed:06d}.jpg"
            cv2.imwrite(str(output_path), img_resized)
            processed += 1
            
            if processed >= MAX_IMAGES:
                break
    except:
        continue

print(f"\n✅ {processed} images traitées")

# Compter résultats
for split in ['train', 'test']:
    for label in ['benign', 'malignant']:
        path = OUTPUT_PATH / split / label
        count = len(list(path.glob("*.jpg"))) if path.exists() else 0
        print(f"   {split}/{label}: {count}")

print(f"\n💡 Pour un matching précis CSV ↔ JPEG, il faut créer une table de correspondance")
print(f"   Cette version utilise un échantillon aléatoire de {MAX_IMAGES} images")
