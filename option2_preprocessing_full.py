"""
OPTION 2 : PREPROCESSING COMPLET 6000 IMAGES
Prétraite toutes les images avec labels corrects depuis CSV

Fonctionnalités :
- Utilise les chemins CSV pour labélisation correcte
- CLAHE, débruitage, normalisation
- Organisation train/test par label
- Augmentation de données (train uniquement)
- Rapport détaillé

Auteur: TIA Ange Jules-Rihem ben Maouia
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "csv"
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned" / "csv"
JPEG_PATH = BASE_DIR / "jpeg"
OUTPUT_PATH = BASE_DIR / "data" / "processed_images_full"
REPORTS_PATH = BASE_DIR / "reports"

# Paramètres
TARGET_SIZE = (224, 224)
APPLY_AUGMENTATION = True  # Sur train seulement
MAX_IMAGES_PER_DATASET = None  # None = tous, ou nombre pour limiter

print("="*80)
print("🖼️  PREPROCESSING COMPLET 6000 IMAGES")
print("="*80 + "\n")

# Créer structure
for split in ['train', 'test']:
    for label in ['benign', 'malignant']:
        (OUTPUT_PATH / split / label).mkdir(parents=True, exist_ok=True)

# ==========================================
# FONCTIONS DE PRÉTRAITEMENT
# ==========================================

def preprocess_image(img_path):
    """Prétraite une image complète"""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Redimensionner
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Débruitage
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        
        # Normalisation
        img_float = img.astype(np.float32)
        mean = np.mean(img_float)
        std = np.std(img_float)
        if std > 0:
            img_normalized = (img_float - mean) / std
        else:
            img_normalized = img_float - mean
        
        img_normalized = np.clip(img_normalized * 255, 0, 255).astype(np.uint8)
        
        return img_normalized
    
    except Exception as e:
        return None

def augment_image(img):
    """Génère 3 versions augmentées"""
    augmented = []
    
    # Rotation ±10°
    for angle in [-10, 10]:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        augmented.append(rotated)
    
    # Flip horizontal
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    
    return augmented

# ==========================================
# TRAITEMENT PAR DATASET
# ==========================================

csv_source = CLEANED_CSV_PATH if CLEANED_CSV_PATH.exists() and list(CLEANED_CSV_PATH.glob("*.csv")) else CSV_PATH

print(f"Source CSV: {csv_source}")
print(f"Source images: {JPEG_PATH}")
print(f"Destination: {OUTPUT_PATH}\n")

processing_report = {
    'timestamp': datetime.now().isoformat(),
    'parameters': {
        'target_size': TARGET_SIZE,
        'augmentation': APPLY_AUGMENTATION,
        'max_per_dataset': MAX_IMAGES_PER_DATASET
    },
    'datasets': {},
    'summary': {
        'total_processed': 0,
        'total_augmented': 0,
        'total_failed': 0
    }
}

csv_files = list(csv_source.glob("*.csv"))

for csv_file in csv_files:
    print(f"\n{'='*80}")
    print(f"📊 {csv_file.name}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(csv_file)
    
    # Déterminer split (train ou test)
    split = 'train' if 'train' in csv_file.name.lower() else 'test'
    
    # Trouver colonne chemin
    path_col = None
    for col in ['image file path', 'image_file_path', 'cropped image file path']:
        if col in df.columns:
            path_col = col
            break
    
    if not path_col or 'pathology' not in df.columns:
        print(f"   ⚠️ Colonnes requises non trouvées")
        continue
    
    # Harmoniser pathology
    df['pathology'] = df['pathology'].astype(str).str.lower().str.strip()
    df['pathology'] = df['pathology'].replace({'benign_without_callback': 'benign'})
    
    # Filtrer benign/malignant
    df = df[df['pathology'].isin(['benign', 'malignant'])]
    
    # Limiter si demandé
    if MAX_IMAGES_PER_DATASET:
        df = df.head(MAX_IMAGES_PER_DATASET)
    
    print(f"Split: {split}")
    print(f"Images à traiter: {len(df)}\n")
    
    stats = {
        'split': split,
        'total_rows': len(df),
        'processed': 0,
        'augmented': 0,
        'failed': 0,
        'by_label': {'benign': 0, 'malignant': 0}
    }
    
    # Traiter chaque image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        label = row['pathology']
        img_path_str = row[path_col]
        
        if pd.isna(img_path_str):
            stats['failed'] += 1
            continue
        
        # Chemin complet
        img_path = JPEG_PATH / img_path_str
        
        if not img_path.exists():
            stats['failed'] += 1
            continue
        
        # Prétraiter
        img_processed = preprocess_image(img_path)
        
        if img_processed is None:
            stats['failed'] += 1
            continue
        
        # Sauvegarder original
        output_filename = f"{split}_{label}_{idx:06d}_original.jpg"
        output_path = OUTPUT_PATH / split / label / output_filename
        cv2.imwrite(str(output_path), img_processed)
        
        stats['processed'] += 1
        stats['by_label'][label] += 1
        processing_report['summary']['total_processed'] += 1
        
        # Augmentation (train uniquement)
        if APPLY_AUGMENTATION and split == 'train':
            augmented = augment_image(img_processed)
            for aug_idx, aug_img in enumerate(augmented, 1):
                aug_filename = f"{split}_{label}_{idx:06d}_aug{aug_idx}.jpg"
                aug_path = OUTPUT_PATH / split / label / aug_filename
                cv2.imwrite(str(aug_path), aug_img)
                stats['augmented'] += 1
                processing_report['summary']['total_augmented'] += 1
    
    # Afficher résultats
    print(f"\n✅ Résultats:")
    print(f"   Traitées: {stats['processed']:,}")
    print(f"   Augmentées: {stats['augmented']:,}")
    print(f"   Échecs: {stats['failed']:,}")
    print(f"   Benign: {stats['by_label']['benign']:,}")
    print(f"   Malignant: {stats['by_label']['malignant']:,}")
    
    processing_report['datasets'][csv_file.name] = stats

# ==========================================
# RÉSUMÉ FINAL
# ==========================================

print(f"\n{'='*80}")
print(f"📋 RÉSUMÉ FINAL")
print(f"{'='*80}\n")

print(f"✅ Images traitées: {processing_report['summary']['total_processed']:,}")
print(f"✅ Images augmentées: {processing_report['summary']['total_augmented']:,}")
print(f"✅ Total: {processing_report['summary']['total_processed'] + processing_report['summary']['total_augmented']:,}")
print(f"❌ Échecs: {processing_report['summary']['total_failed']:,}")

# Compter fichiers générés
train_benign = len(list((OUTPUT_PATH / 'train' / 'benign').glob("*.jpg"))) if (OUTPUT_PATH / 'train' / 'benign').exists() else 0
train_malignant = len(list((OUTPUT_PATH / 'train' / 'malignant').glob("*.jpg"))) if (OUTPUT_PATH / 'train' / 'malignant').exists() else 0
test_benign = len(list((OUTPUT_PATH / 'test' / 'benign').glob("*.jpg"))) if (OUTPUT_PATH / 'test' / 'benign').exists() else 0
test_malignant = len(list((OUTPUT_PATH / 'test' / 'malignant').glob("*.jpg"))) if (OUTPUT_PATH / 'test' / 'malignant').exists() else 0

print(f"\n📂 Structure finale:")
print(f"   train/benign: {train_benign:,}")
print(f"   train/malignant: {train_malignant:,}")
print(f"   test/benign: {test_benign:,}")
print(f"   test/malignant: {test_malignant:,}")

# Sauvegarder rapport
report_path = REPORTS_PATH / 'preprocessing_full_images.json'
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(processing_report, f, indent=4, default=str)

print(f"\n💾 Rapport sauvegardé: {report_path}")
print(f"\n✨ PREPROCESSING TERMINÉ!")
print(f"{'='*80}\n")
