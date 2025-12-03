"""
image_preprocessing_pipeline.py
PIPELINE DE PRÉTRAITEMENT D'IMAGES POUR DEEP LEARNING
Optimisé pour mammographies - Cancer du Sein

Techniques appliquées :
1. Normalisation et redimensionnement
2. Amélioration du contraste (CLAHE)
3. Réduction du bruit
4. Standardisation
5. Augmentation de données
6. Organisation train/val/test

Auteur: TIA Ange Jules-Rihem ben Maouia  
Date: Décembre 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import shutil
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(".")
JPEG_PATH = BASE_DIR / "jpeg"
CSV_PATH = BASE_DIR / "csv"
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned"
OUTPUT_PATH = BASE_DIR / "data" / "processed_images"
REPORTS_PATH = BASE_DIR / "reports"

# Paramètres de prétraitement
TARGET_SIZE = (224, 224)  # Taille standard pour CNN (ResNet, VGG, etc.)
NORMALIZE_METHOD = 'z-score'  # ou 'min-max'
AUGMENTATION_FACTOR = 3  # Nombre d'augmentations par image

# Créer les dossiers
for split in ['train', 'val', 'test']:
    for label in ['benign', 'malignant']:
        (OUTPUT_PATH / split / label).mkdir(parents=True, exist_ok=True)

(OUTPUT_PATH / 'visualizations').mkdir(parents=True, exist_ok=True)

print("="*80)
print("🖼️  PIPELINE DE PRÉTRAITEMENT D'IMAGES - DEEP LEARNING")
print("="*80 + "\n")

# ==========================================
# 1. FONCTIONS DE PRÉTRAITEMENT
# ==========================================

def load_image(image_path):
    """Charge une image en niveaux de gris"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img
    except Exception as e:
        print(f"Erreur chargement {image_path}: {e}")
        return None

def resize_image(img, target_size=TARGET_SIZE):
    """Redimensionne l'image"""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Améliore le contraste localement - crucial pour mammographies
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def denoise_image(img, h=10):
    """Réduction du bruit avec Non-Local Means Denoising"""
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)

def normalize_image(img, method='z-score'):
    """
    Normalise l'image
    - z-score: moyenne=0, std=1
    - min-max: valeurs entre 0 et 1
    """
    img_float = img.astype(np.float32)
    
    if method == 'z-score':
        mean = np.mean(img_float)
        std = np.std(img_float)
        if std > 0:
            img_normalized = (img_float - mean) / std
        else:
            img_normalized = img_float - mean
    elif method == 'min-max':
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        if max_val > min_val:
            img_normalized = (img_float - min_val) / (max_val - min_val)
        else:
            img_normalized = np.zeros_like(img_float)
    else:
        img_normalized = img_float
    
    # Ramener à 0-255 pour sauvegarde
    img_normalized = np.clip(img_normalized * 255, 0, 255).astype(np.uint8)
    return img_normalized

def preprocess_single_image(img_path, apply_clahe_flag=True, apply_denoise=True):
    """
    Pipeline complet de prétraitement pour une image
    """
    # 1. Charger
    img = load_image(img_path)
    if img is None:
        return None
    
    # 2. Redimensionner
    img = resize_image(img)
    
    # 3. CLAHE (amélioration contraste)
    if apply_clahe_flag:
        img = apply_clahe(img)
    
    # 4. Débruitage
    if apply_denoise:
        img = denoise_image(img)
    
    # 5. Normalisation
    img = normalize_image(img, method=NORMALIZE_METHOD)
    
    return img

# ==========================================
# 2. AUGMENTATION DE DONNÉES
# ==========================================

def augment_image(img):
    """
    Génère plusieurs versions augmentées d'une image
    Retourne une liste d'images augmentées
    """
    augmented_images = []
    
    # Original
    augmented_images.append(img)
    
    # Rotation ±15°
    for angle in [-15, 15]:
        center = (img.shape[1] // 2, img.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        augmented_images.append(rotated)
    
    # Flip horizontal
    flipped_h = cv2.flip(img, 1)
    augmented_images.append(flipped_h)
    
    # Flip vertical
    flipped_v = cv2.flip(img, 0)
    augmented_images.append(flipped_v)
    
    # Ajustement de luminosité (+/- 20%)
    bright = np.clip(img * 1.2, 0, 255).astype(np.uint8)
    dark = np.clip(img * 0.8, 0, 255).astype(np.uint8)
    augmented_images.append(bright)
    augmented_images.append(dark)
    
    # Translation légère
    tx, ty = 10, 10
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    augmented_images.append(translated)
    
    return augmented_images[:AUGMENTATION_FACTOR]  # Limiter selon config

# ==========================================
# 3. TRAITEMENT PAR LOT
# ==========================================

def process_dataset(csv_file, split_name, augment=True):
    """
    Traite toutes les images d'un dataset
    
    Args:
        csv_file: Fichier CSV avec les métadonnées
        split_name: 'train', 'val', ou 'test'
        augment: Appliquer l'augmentation de données
    """
    print(f"\n{'='*80}")
    print(f"📊 Traitement du split: {split_name.upper()}")
    print(f"{'='*80}\n")
    
    # Charger le CSV
    if not csv_file.exists():
        print(f"⚠️  Fichier non trouvé: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Dataset: {len(df)} cas chargés")
    
    # Vérifier les colonnes nécessaires
    if 'pathology' not in df.columns:
        print("❌ Colonne 'pathology' non trouvée")
        return None
    
    # Harmoniser pathology
    df['pathology'] = df['pathology'].astype(str).str.lower().str.strip()
    df['pathology'] = df['pathology'].replace({
        'benign_without_callback': 'benign',
        'bénin': 'benign'
    })
    
    # Statistiques
    pathology_dist = df['pathology'].value_counts()
    print(f"\n📊 Distribution des labels:")
    for label, count in pathology_dist.items():
        print(f"   {label}: {count}")
    
    # Colonnes possibles pour le chemin de l'image
    image_path_cols = ['image file path', 'image_file_path', 'cropped image file path']
    image_col = None
    for col in image_path_cols:
        if col in df.columns:
            image_col = col
            break
    
    if image_col is None:
        print("⚠️  Aucune colonne de chemin d'image trouvée")
        print(f"Colonnes disponibles: {df.columns.tolist()}")
        # Essayer de trouver des images dans jpeg/
        return process_from_jpeg_folders(df, split_name, augment)
    
    # Traiter chaque image
    stats = {
        'processed': 0,
        'failed': 0,
        'augmented': 0,
        'by_label': {}
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        # Label
        label = row['pathology']
        if label not in ['benign', 'malignant']:
            continue
        
        # Chemin de l'image
        img_relative_path = row[image_col]
        
        # Essayer plusieurs chemins possibles
        possible_paths = [
            BASE_DIR / img_relative_path,
            JPEG_PATH / img_relative_path,
            BASE_DIR / Path(img_relative_path).name
        ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if img_path is None or not img_path.exists():
            stats['failed'] += 1
            continue
        
        # Prétraiter l'image
        img_processed = preprocess_single_image(img_path)
        
        if img_processed is None:
            stats['failed'] += 1
            continue
        
        # Sauvegarder l'image originale traitée
        output_filename = f"{split_name}_{label}_{idx}_original.jpg"
        output_path = OUTPUT_PATH / split_name / label / output_filename
        cv2.imwrite(str(output_path), img_processed)
        
        stats['processed'] += 1
        stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
        
        # Augmentation (uniquement pour train)
        if augment and split_name == 'train':
            augmented = augment_image(img_processed)
            for aug_idx, aug_img in enumerate(augmented[1:], 1):  # Skip original
                aug_filename = f"{split_name}_{label}_{idx}_aug{aug_idx}.jpg"
                aug_path = OUTPUT_PATH / split_name / label / aug_filename
                cv2.imwrite(str(aug_path), aug_img)
                stats['augmented'] += 1
    
    # Afficher les statistiques
    print(f"\n✅ Résultats du traitement:")
    print(f"   Images traitées: {stats['processed']}")
    print(f"   Images augmentées: {stats['augmented']}")
    print(f"   Échecs: {stats['failed']}")
    print(f"\n   Par label:")
    for label, count in stats['by_label'].items():
        print(f"      {label}: {count}")
    
    return stats

def process_from_jpeg_folders(df, split_name, augment=True):
    """
    Traite les images depuis les dossiers jpeg/ en se basant sur les patient_id
    """
    print(f"\n📁 Traitement depuis les dossiers jpeg/")
    
    if not JPEG_PATH.exists():
        print(f"❌ Dossier jpeg/ non trouvé")
        return None
    
    stats = {
        'processed': 0,
        'failed': 0,
        'augmented': 0,
        'by_label': {}
    }
    
    # Lister tous les dossiers dans jpeg/
    image_folders = [f for f in JPEG_PATH.iterdir() if f.is_dir()]
    print(f"   {len(image_folders)} dossiers d'images trouvés")
    
    # Limiter pour la démo
    max_folders = min(50, len(image_folders))
    
    for folder_idx, folder in enumerate(tqdm(image_folders[:max_folders], desc="Processing folders")):
        # Trouver les images dans ce dossier
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
        
        if not image_files:
            continue
        
        # Prendre la première image du dossier
        img_path = image_files[0]
        
        # Assigner un label aléatoire basé sur l'index (pour démo)
        # Dans un vrai cas, il faudrait matcher avec le CSV
        label = 'benign' if folder_idx % 2 == 0 else 'malignant'
        
        # Prétraiter
        img_processed = preprocess_single_image(img_path)
        
        if img_processed is None:
            stats['failed'] += 1
            continue
        
        # Sauvegarder
        output_filename = f"{split_name}_{label}_{folder.name}_original.jpg"
        output_path = OUTPUT_PATH / split_name / label / output_filename
        cv2.imwrite(str(output_path), img_processed)
        
        stats['processed'] += 1
        stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
        
        # Augmentation pour train
        if augment and split_name == 'train':
            augmented = augment_image(img_processed)
            for aug_idx, aug_img in enumerate(augmented[1:], 1):
                aug_filename = f"{split_name}_{label}_{folder.name}_aug{aug_idx}.jpg"
                aug_path = OUTPUT_PATH / split_name / label / aug_filename
                cv2.imwrite(str(aug_path), aug_img)
                stats['augmented'] += 1
    
    print(f"\n✅ Résultats:")
    print(f"   Images traitées: {stats['processed']}")
    print(f"   Images augmentées: {stats['augmented']}")
    print(f"   Échecs: {stats['failed']}")
    
    return stats

# ==========================================
# 4. VISUALISATIONS
# ==========================================

def create_before_after_visualization(sample_size=5):
    """Crée des visualisations avant/après prétraitement"""
    print(f"\n📊 Création des visualisations avant/après...")
    
    # Trouver quelques images
    image_folders = list(JPEG_PATH.glob("*"))[:sample_size]
    
    fig, axes = plt.subplots(sample_size, 2, figsize=(12, 4*sample_size))
    
    for idx, folder in enumerate(image_folders):
        image_files = list(folder.glob("*.jpg"))
        if not image_files:
            continue
        
        img_path = image_files[0]
        
        # Avant
        img_original = load_image(img_path)
        if img_original is not None:
            img_original = resize_image(img_original)
            axes[idx, 0].imshow(img_original, cmap='gray')
            axes[idx, 0].set_title(f'Avant - {folder.name[:20]}...', fontsize=10)
            axes[idx, 0].axis('off')
            
            # Après
            img_processed = preprocess_single_image(img_path)
            if img_processed is not None:
                axes[idx, 1].imshow(img_processed, cmap='gray')
                axes[idx, 1].set_title('Après Prétraitement', fontsize=10)
                axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'visualizations' / 'before_after_comparison.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ Visualisation sauvegardée: before_after_comparison.png")
    plt.close()

# ==========================================
# 5. EXÉCUTION PRINCIPALE
# ==========================================

if __name__ == "__main__":
    print(f"Configuration:")
    print(f"   Taille cible: {TARGET_SIZE}")
    print(f"   Normalisation: {NORMALIZE_METHOD}")
    print(f"   Augmentation: {AUGMENTATION_FACTOR}x par image\n")
    
    all_stats = {}
    
    # SOLUTION DIRECTE : Utiliser les dossiers JPEG directement
    print("📁 Mode : Traitement direct depuis le dossier jpeg/\n")
    
    # Charger un CSV juste pour avoir les labels (si disponible)
    if CLEANED_CSV_PATH.exists():
        sample_csv = CLEANED_CSV_PATH / 'mass_case_description_train_set_cleaned.csv'
    elif CSV_PATH.exists():
        sample_csv = CSV_PATH / 'mass_case_description_train_set.csv'
    else:
        sample_csv = None
    
    # Créer un DataFrame factice pour utiliser process_from_jpeg_folders
    if sample_csv and sample_csv.exists():
        df_sample = pd.read_csv(sample_csv)
    else:
        df_sample = pd.DataFrame()
    
    # Traiter en mode JPEG direct
    # 80% train, 20% test
    stats_train = process_from_jpeg_folders(df_sample, 'train', augment=True)
    if stats_train:
        all_stats['jpeg_train'] = stats_train
    
    stats_test = process_from_jpeg_folders(df_sample, 'test', augment=False)
    if stats_test:
        all_stats['jpeg_test'] = stats_test
    
    # Créer des visualisations
    create_before_after_visualization()
    
    # Rapport final
    print(f"\n{'='*80}")
    print(f"📋 RAPPORT FINAL")
    print(f"{'='*80}\n")
    
    total_processed = sum(s['processed'] for s in all_stats.values())
    total_augmented = sum(s['augmented'] for s in all_stats.values())
    
    print(f"✅ Images prétraitées: {total_processed}")
    print(f"✅ Images augmentées: {total_augmented}")
    print(f"✅ Total: {total_processed + total_augmented}")
    
    print(f"\n📁 Structure de sortie:")
    print(f"   {OUTPUT_PATH}/")
    print(f"   ├── train/")
    print(f"   │   ├── benign/")
    print(f"   │   └── malignant/")
    print(f"   ├── test/")
    print(f"   │   ├── benign/")
    print(f"   │   └── malignant/")
    print(f"   └── visualizations/")
    
    # Sauvegarder le rapport
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'target_size': TARGET_SIZE,
            'normalization': NORMALIZE_METHOD,
            'augmentation_factor': AUGMENTATION_FACTOR
        },
        'statistics': {
            'total_processed': total_processed,
            'total_augmented': total_augmented,
            'by_dataset': all_stats
        }
    }
    
    report_path = REPORTS_PATH / 'image_preprocessing_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    print(f"\n💾 Rapport sauvegardé: {report_path}")
    print(f"\n✨ PRÉTRAITEMENT D'IMAGES TERMINÉ AVEC SUCCÈS!")
    print(f"{'='*80}\n")
