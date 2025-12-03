"""
data_augmentation.py
AUGMENTATION DES DONNÉES - Dataset Cancer du Sein
- Augmentation des images (transformations géométriques)
- Augmentation des données tabulaires (SMOTE)
- Génération de rapports avant/après
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import os
import json
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Pour l'augmentation tabulaire
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

# Pour l'augmentation d'images
from torchvision import transforms

# Configuration
BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "csv"
IMAGE_PATH = BASE_DIR / "jpeg"
REPORTS_PATH = BASE_DIR / "reports"
AUGMENTED_PATH = BASE_DIR / "data" / "augmented"

# Créer les dossiers
AUGMENTED_PATH.mkdir(parents=True, exist_ok=True)
(AUGMENTED_PATH / "images").mkdir(exist_ok=True)
(AUGMENTED_PATH / "csv").mkdir(exist_ok=True)

print("="*80)
print("📈 AUGMENTATION DES DONNÉES - CANCER DU SEIN")
print("="*80 + "\n")

# ==========================================
# 1. AUGMENTATION D'IMAGES
# ==========================================

print("🖼️  PARTIE 1: AUGMENTATION DES IMAGES")
print("="*80 + "\n")

# Définir les transformations d'augmentation
augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
])

def augment_image(image_path, output_dir, num_augmentations=3):
    """
    Augmente une image avec différentes transformations
    
    Args:
        image_path: Chemin de l'image source
        output_dir: Dossier de sortie
        num_augmentations: Nombre d'images augmentées à générer
    
    Returns:
        list: Chemins des images générées
    """
    try:
        img = Image.open(image_path).convert('RGB')
        generated_paths = []
        
        # Sauvegarder l'original
        original_name = Path(image_path).stem
        original_ext = Path(image_path).suffix
        
        for i in range(num_augmentations):
            # Appliquer les transformations
            img_augmented = augmentation_transforms(img)
            
            # Sauvegarder
            output_filename = f"{original_name}_aug{i+1}{original_ext}"
            output_path = Path(output_dir) / output_filename
            img_augmented.save(output_path, quality=95)
            generated_paths.append(output_path)
        
        return generated_paths
    
    except Exception as e:
        print(f"   ❌ Erreur sur {image_path}: {e}")
        return []

def augment_image_dataset(source_dir, target_dir, max_folders=5, 
                          images_per_folder=10, augmentations_per_image=3):
    """
    Augmente un dataset d'images complet
    
    Args:
        source_dir: Dossier source des images
        target_dir: Dossier de destination
        max_folders: Nombre maximum de dossiers à traiter
        images_per_folder: Nombre d'images à augmenter par dossier
        augmentations_per_image: Nombre d'augmentations par image
    """
    print(f"📂 Augmentation des images depuis: {source_dir}")
    print(f"   Cible: {target_dir}")
    print(f"   Paramètres: {images_per_folder} images × {augmentations_per_image} augmentations\n")
    
    if not os.path.exists(source_dir):
        print(f"❌ Dossier source introuvable: {source_dir}")
        return {'folders_processed': 0, 'original_images': 0, 'augmented_images': 0}
    
    # Lister les dossiers d'images
    image_folders = [f for f in os.listdir(source_dir) 
                    if os.path.isdir(os.path.join(source_dir, f))]
    
    image_folders = image_folders[:max_folders]
    
    stats = {
        'folders_processed': 0,
        'original_images': 0,
        'augmented_images': 0
    }
    
    for folder in image_folders:
        folder_path = os.path.join(source_dir, folder)
        output_folder = os.path.join(target_dir, folder)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n📁 Traitement du dossier: {folder}")
        
        # Lister les images
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limiter le nombre d'images
        image_files = image_files[:images_per_folder]
        
        print(f"   {len(image_files)} images sélectionnées pour augmentation")
        
        for img_file in tqdm(image_files, desc="   Augmentation"):
            img_path = os.path.join(folder_path, img_file)
            
            # Augmenter l'image
            generated = augment_image(img_path, output_folder, augmentations_per_image)
            
            stats['original_images'] += 1
            stats['augmented_images'] += len(generated)
        
        stats['folders_processed'] += 1
        print(f"   ✅ {stats['augmented_images']} images augmentées générées")
    
    return stats

# Exécuter l'augmentation d'images (sur un échantillon)
if os.path.exists(IMAGE_PATH):
    print("\n🚀 Démarrage de l'augmentation d'images...\n")
    image_stats = augment_image_dataset(
        source_dir=IMAGE_PATH,
        target_dir=AUGMENTED_PATH / "images",
        max_folders=3,  # Limiter pour la démo
        images_per_folder=10,
        augmentations_per_image=3
    )
    
    print(f"\n📊 Statistiques d'augmentation d'images:")
    print(f"   Dossiers traités: {image_stats['folders_processed']}")
    print(f"   Images originales: {image_stats['original_images']}")
    print(f"   Images augmentées générées: {image_stats['augmented_images']}")
    if image_stats['original_images'] > 0:
        print(f"   Facteur d'augmentation: ×{image_stats['augmented_images']/image_stats['original_images']:.1f}")
else:
    print(f"⚠️ Dossier images introuvable: {IMAGE_PATH}")
    print("   Passage à l'augmentation tabulaire...")
    image_stats = {'folders_processed': 0, 'original_images': 0, 'augmented_images': 0}

# Visualiser des exemples d'augmentation
def visualize_augmentation_examples(source_dir, n_examples=3):
    """Visualise des exemples d'augmentation"""
    
    # Trouver quelques images
    all_images = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
                if len(all_images) >= n_examples:
                    break
        if len(all_images) >= n_examples:
            break
    
    if not all_images:
        print("⚠️ Aucune image trouvée pour visualisation")
        return
    
    fig, axes = plt.subplots(n_examples, 4, figsize=(16, 4*n_examples))
    
    for idx, img_path in enumerate(all_images[:n_examples]):
        # Image originale
        img = Image.open(img_path).convert('RGB')
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Original', fontweight='bold')
        axes[idx, 0].axis('off')
        
        # 3 augmentations
        for aug_idx in range(3):
            img_aug = augmentation_transforms(img)
            axes[idx, aug_idx+1].imshow(img_aug)
            axes[idx, aug_idx+1].set_title(f'Augmentation {aug_idx+1}', fontweight='bold')
            axes[idx, aug_idx+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / 'augmentation_examples_images.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Exemples d'augmentation sauvegardés: {REPORTS_PATH / 'augmentation_examples_images.png'}")
    plt.close()

if os.path.exists(AUGMENTED_PATH / "images") and image_stats['augmented_images'] > 0:
    visualize_augmentation_examples(AUGMENTED_PATH / "images")

# ==========================================
# 2. AUGMENTATION DES DONNÉES TABULAIRES
# ==========================================

print(f"\n\n{'='*80}")
print("📊 PARTIE 2: AUGMENTATION DES DONNÉES TABULAIRES (SMOTE)")
print("="*80 + "\n")

# Chercher les datasets nettoyés ou originaux
cleaned_path = BASE_DIR / "data" / "cleaned"
if cleaned_path.exists():
    cleaned_files = list(cleaned_path.glob('*.csv'))
    print(f"✅ {len(cleaned_files)} fichiers nettoyés trouvés")
else:
    cleaned_files = []

if not cleaned_files:
    print("⚠️ Aucun fichier nettoyé trouvé. Utilisation des fichiers originaux...")
    cleaned_files = [
        DATA_PATH / 'calc_case_description_train_set.csv',
        DATA_PATH / 'mass_case_description_train_set.csv'
    ]
    cleaned_files = [f for f in cleaned_files if f.exists()]

tabular_augmentation_stats = []

for csv_file in cleaned_files:
    if not os.path.exists(csv_file):
        continue
    
    print(f"\n{'─'*80}")
    print(f"📄 Traitement de: {csv_file.name}")
    print(f"{'─'*80}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"   Dataset original: {len(df)} lignes × {df.shape[1]} colonnes")
        
        # Identifier la colonne cible (généralement celle avec 'pathology' dans le nom)
        target_col = None
        for col in df.columns:
            if 'pathology' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
                target_col = col
                break
        
        if target_col is None:
            print("   ⚠️ Aucune colonne cible identifiée. Passage au fichier suivant...")
            continue
        
        print(f"   Colonne cible identifiée: {target_col}")
        
        # Vérifier la distribution des classes
        class_dist = df[target_col].value_counts()
        print(f"\n   📊 Distribution des classes (avant):")
        for cls, count in class_dist.items():
            print(f"      {cls}: {count}")
        
        # Vérifier le déséquilibre
        if len(class_dist) < 2:
            print("   ⚠️ Une seule classe détectée. Pas d'augmentation possible.")
            continue
            
        imbalance_ratio = class_dist.max() / class_dist.min()
        print(f"\n   ⚖️  Ratio de déséquilibre: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio < 1.5:
            print(f"   ✅ Dataset équilibré. Pas d'augmentation nécessaire.")
            continue
        
        # Préparer les données pour SMOTE
        # Séparer features et target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encoder les variables catégorielles
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Gérer les colonnes catégorielles dans X
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        label_encoders = {}
        for col in categorical_cols:
            le_col = LabelEncoder()
            X_encoded[col] = le_col.fit_transform(X[col].astype(str))
            label_encoders[col] = le_col
        
        # Gérer les valeurs manquantes
        X_encoded = X_encoded.fillna(X_encoded.median())
        
        # Standardiser
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        # Appliquer SMOTE
        print(f"\n   🔄 Application de SMOTE...")
        k_neighbors = min(5, class_dist.min()-1)
        if k_neighbors < 1:
            print(f"   ⚠️ Pas assez d'échantillons pour SMOTE (min: {class_dist.min()})")
            continue
            
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)
            
            # Statistiques après augmentation
            unique, counts = np.unique(y_resampled, return_counts=True)
            class_dist_after = dict(zip(le.inverse_transform(unique), counts))
            
            print(f"\n   📊 Distribution des classes (après):")
            for cls, count in class_dist_after.items():
                print(f"      {cls}: {count}")
            
            print(f"\n   ✅ Augmentation réussie!")
            print(f"      Lignes avant: {len(df)}")
            print(f"      Lignes après: {len(X_resampled)}")
            print(f"      Facteur d'augmentation: ×{len(X_resampled)/len(df):.2f}")
            
            # Recréer un DataFrame
            X_resampled_df = pd.DataFrame(
                scaler.inverse_transform(X_resampled),
                columns=X_encoded.columns
            )
            
            # Décoder les variables catégorielles
            for col, le_col in label_encoders.items():
                X_resampled_df[col] = le_col.inverse_transform(
                    X_resampled_df[col].round().astype(int)
                )
            
            # Ajouter la colonne cible
            X_resampled_df[target_col] = le.inverse_transform(y_resampled)
            
            # Sauvegarder
            output_filename = csv_file.stem.replace('_cleaned', '').replace('_train_set', '') + '_augmented.csv'
            output_path = AUGMENTED_PATH / "csv" / output_filename
            X_resampled_df.to_csv(output_path, index=False)
            print(f"   💾 Dataset augmenté sauvegardé: {output_path}")
            
            # Sauvegarder les stats
            tabular_augmentation_stats.append({
                'file': csv_file.name,
                'rows_before': len(df),
                'rows_after': len(X_resampled),
                'augmentation_factor': len(X_resampled)/len(df)
            })
            
            # Visualisation avant/après
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Avant
            ax1 = axes[0]
            class_dist.plot(kind='bar', ax=ax1, color='coral')
            ax1.set_title('Distribution Avant Augmentation', fontweight='bold')
            ax1.set_xlabel('Classe')
            ax1.set_ylabel('Nombre d\'échantillons')
            ax1.tick_params(axis='x', rotation=45)
            
            # Après
            ax2 = axes[1]
            pd.Series(class_dist_after).plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title('Distribution Après Augmentation (SMOTE)', fontweight='bold')
            ax2.set_xlabel('Classe')
            ax2.set_ylabel('Nombre d\'échantillons')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            viz_filename = csv_file.stem.replace('_cleaned', '').replace('_train_set', '') + '_augmentation_viz.png'
            plt.savefig(REPORTS_PATH / viz_filename, dpi=300, bbox_inches='tight')
            print(f"   💾 Visualisation sauvegardée: {REPORTS_PATH / viz_filename}")
            plt.close()
            
        except Exception as e:
            print(f"   ❌ Erreur lors de l'application de SMOTE: {e}")
            continue
    
    except Exception as e:
        print(f"   ❌ Erreur lors du traitement: {e}")
        continue

# ==========================================
# 3. RAPPORT RÉCAPITULATIF D'AUGMENTATION
# ==========================================

print(f"\n\n{'='*80}")
print("📋 RAPPORT RÉCAPITULATIF D'AUGMENTATION")
print("="*80 + "\n")

# Compter les fichiers générés
n_images_augmented = len(list((AUGMENTED_PATH / "images").rglob('*.jpg'))) if (AUGMENTED_PATH / "images").exists() else 0
n_csv_augmented = len(list((AUGMENTED_PATH / "csv").glob('*_augmented.csv'))) if (AUGMENTED_PATH / "csv").exists() else 0

print(f"✅ Résultats de l'augmentation:")
print(f"   Images augmentées: {n_images_augmented}")
print(f"   Datasets CSV augmentés: {n_csv_augmented}")

# Créer un rapport JSON
augmentation_report = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'image_augmentation': {
        'total_images_generated': n_images_augmented,
        'source_directory': str(IMAGE_PATH),
        'output_directory': str(AUGMENTED_PATH / "images"),
        'stats': image_stats
    },
    'tabular_augmentation': {
        'method': 'SMOTE',
        'datasets_augmented': n_csv_augmented,
        'output_directory': str(AUGMENTED_PATH / "csv"),
        'details': tabular_augmentation_stats
    }
}

with open(REPORTS_PATH / 'augmentation_report.json', 'w') as f:
    json.dump(augmentation_report, f, indent=4)

print(f"\n💾 Rapport d'augmentation sauvegardé: {REPORTS_PATH / 'augmentation_report.json'}")

print(f"\n📁 Structure des fichiers augmentés:")
print(f"   {AUGMENTED_PATH}/")
print(f"   ├── images/")
print(f"   │   └── {image_stats['folders_processed']} dossiers avec {n_images_augmented} images augmentées")
print(f"   └── csv/")
print(f"       └── {n_csv_augmented} fichiers CSV augmentés")

print(f"\n🎯 PROCHAINES ÉTAPES:")
print("""
1. ✅ Exploration des données terminée
2. ✅ Nettoyage des données effectué
3. ✅ Augmentation des données réalisée
4. ⏳ Feature engineering (optionnel)
5. ⏳ Entraînement de modèles (optionnel)
6. ⏳ Finalisation du rapport
""")

print(f"\n{'='*80}")
print("✨ AUGMENTATION DES DONNÉES TERMINÉE AVEC SUCCÈS!")
print(f"{'='*80}\n")
