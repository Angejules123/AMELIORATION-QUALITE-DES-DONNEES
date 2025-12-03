"""
CONSOLIDATION FINALE - VISUALISATION ET QUALITÉ AVANCÉE
Crée visualisations étiquetées et améliore la qualité des images

Fonctionnalités :
1. Visualisation mosaïque avec labels
2. Amélioration qualité avancée (sharpening, edge enhancement)
3. Comparaisons avant/après
4. Nettoyage anciens résultats
5. Export organisé

Auteur: TIA Ange Jules-Rihem ben Maouia
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import shutil
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "csv"
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned" / "csv"
JPEG_PATH = BASE_DIR / "jpeg"
PROCESSED_PATH = BASE_DIR / "data" / "processed_images_full"
FINAL_OUTPUT = BASE_DIR / "data" / "final_dataset"
FIGURES_PATH = BASE_DIR / "presentation" / "figures_final"
REPORTS_PATH = BASE_DIR / "reports"

# Créer dossiers
FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🎨 CONSOLIDATION FINALE - VISUALISATION ET QUALITÉ")
print("="*80 + "\n")

# ==========================================
# 1. NETTOYAGE ANCIENS RÉSULTATS
# ==========================================

print("🧹 Nettoyage des anciens résultats")
print("-" * 80)

# Lister dossiers à nettoyer (sauf jpeg, csv, cleaned)
old_folders = [
    BASE_DIR / "data" / "processed_images",  # Ancien
    BASE_DIR / "data" / "augmented",  # Ancien
]

cleaned_count = 0
for folder in old_folders:
    if folder.exists():
        try:
            shutil.rmtree(folder)
            print(f"   ✅ Supprimé: {folder.name}")
            cleaned_count += 1
        except Exception as e:
            print(f"   ⚠️ Erreur sur {folder.name}: {e}")

print(f"   Total nettoyé: {cleaned_count} dossiers\n")

# ==========================================
# 2. CHARGEMENT DATASET ET LABELS
# ==========================================

print("📊 Chargement des données et labels")
print("-" * 80)

csv_source = CLEANED_CSV_PATH if CLEANED_CSV_PATH.exists() else CSV_PATH
csv_files = list(csv_source.glob("*.csv"))

all_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['source_file'] = csv_file.name
    df['split'] = 'train' if 'train' in csv_file.name.lower() else 'test'
    all_data.append(df)

df_combined = pd.concat(all_data, ignore_index=True)

# Harmoniser pathology
if 'pathology' in df_combined.columns:
    df_combined['pathology'] = df_combined['pathology'].astype(str).str.lower().str.strip()
    df_combined['pathology'] = df_combined['pathology'].replace({'benign_without_callback': 'benign'})

print(f"   Total lignes: {len(df_combined):,}")
print(f"   Distribution: {df_combined['pathology'].value_counts().to_dict()}\n")

# ==========================================
# 3. FONCTIONS AMÉLIORATION QUALITÉ
# ==========================================

def enhance_image_quality(img):
    """
    Applique techniques avancées d'amélioration qualité
    """
    # 1. CLAHE adaptatif
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # 2. Débruitage
    img_denoised = cv2.fastNlMeansDenoising(img_clahe, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 3. Sharpening
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    img_sharpened = cv2.filter2D(img_denoised, -1, kernel_sharpen)
    
    # 4. Edge enhancement
    edges = cv2.Canny(img_sharpened, 50, 150)
    img_enhanced = cv2.addWeighted(img_sharpened, 0.9, edges, 0.1, 0)
    
    # 5. Normalisation finale
    img_normalized = cv2.normalize(img_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_normalized

# ==========================================
# 4. VISUALISATIONS ÉTIQUETÉES
# ==========================================

print("🎨 Création visualisations étiquetées")
print("-" * 80)

# Trouver colonne image
path_col = None
for col in ['image file path', 'image_file_path', 'cropped image file path']:
    if col in df_combined.columns:
        path_col = col
        break

if not path_col:
    print("   ⚠️ Colonne image non trouvée")
else:
    # Viz 1: Mosaïque avec labels (8x8 = 64 images)
    fig, axes = plt.subplots(8, 8, figsize=(24, 24))
    fig.suptitle('Dataset Complet - Images Étiquetées', fontsize=20, fontweight='bold', y=0.995)
    
    # Échantillonner images
    sample_size = 64
    df_sample = df_combined[df_combined['pathology'].isin(['benign', 'malignant'])].sample(
        min(sample_size, len(df_combined)), random_state=42
    )
    
    for idx, (ax, (_, row)) in enumerate(zip(axes.flat, df_sample.iterrows())):
        img_path_str = row[path_col]
        label = row['pathology']
        
        if not pd.isna(img_path_str):
            img_path = JPEG_PATH / img_path_str
            
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Redimensionner
                    img = cv2.resize(img, (224, 224))
                    
                    # Afficher
                    ax.imshow(img, cmap='gray')
                    
                    # Label avec couleur
                    color = '#2ecc71' if label == 'benign' else '#e74c3c'
                    ax.set_title(label.upper(), fontsize=10, fontweight='bold', 
                               color='white', backgroundcolor=color, pad=3)
                    ax.axis('off')
                    continue
        
        ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(FIGURES_PATH / '09_mosaique_etiquetee_complete.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ 09_mosaique_etiquetee_complete.png")
    plt.close()
    
    # Viz 2: Comparaison Avant/Après amélioration (6 exemples)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Amélioration Qualité - Avant vs Après', fontsize=18, fontweight='bold')
    
    sample_improve = df_sample.head(6)
    
    for idx, (_, row) in enumerate(sample_improve.iterrows()):
        img_path_str = row[path_col]
        label = row['pathology']
        
        if not pd.isna(img_path_str):
            img_path = JPEG_PATH / img_path_str
            
            if img_path.exists():
                img_original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img_original is not None:
                    img_original = cv2.resize(img_original, (224, 224))
                    img_enhanced = enhance_image_quality(img_original)
                    
                    row_idx = idx // 2
                    col_before = (idx % 2) * 2
                    col_after = col_before + 1
                    
                    # Avant
                    axes[row_idx, col_before].imshow(img_original, cmap='gray')
                    axes[row_idx, col_before].set_title(f'{label.upper()} - Avant', fontweight='bold')
                    axes[row_idx, col_before].axis('off')
                    
                    # Après
                    axes[row_idx, col_after].imshow(img_enhanced, cmap='gray')
                    axes[row_idx, col_after].set_title(f'{label.upper()} - Après', fontweight='bold', color='green')
                    axes[row_idx, col_after].axis('off')
    
    plt.tight_layout()
    fig.savefig(FIGURES_PATH / '10_amelioration_qualite_comparaison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ 10_amelioration_qualite_comparaison.png")
    plt.close()
    
    # Viz 3: Distribution par split et label
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart train/test
    split_counts = df_combined['split'].value_counts()
    axes[0].pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%',
               startangle=90, colors=['#3498db', '#e74c3c'], textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Distribution Train/Test', fontsize=14, fontweight='bold')
    
    # Bar chart label par split
    split_label_counts = df_combined.groupby(['split', 'pathology']).size().unstack(fill_value=0)
    split_label_counts.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
    axes[1].set_title('Distribution Labels par Split', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Split')
    axes[1].legend(title='Pathology', title_fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_PATH / '11_distribution_complete.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ 11_distribution_complete.png\n")
    plt.close()

# ==========================================
# 5. TRAITEMENT FINAL DATASET
# ==========================================

print("💾 Traitement et sauvegarde dataset final")
print("-" * 80)

# Organiser structure finale
for split in ['train', 'test']:
    for label in ['benign', 'malignant']:
        (FINAL_OUTPUT / 'images' / split / label).mkdir(parents=True, exist_ok=True)

# Utiliser images prétraitées si disponibles, sinon originales
if PROCESSED_PATH.exists():
    print(f"   Utilisation images prétraitées depuis {PROCESSED_PATH}")
    source_images = PROCESSED_PATH
else:
    print(f"   Utilisation images originales depuis {JPEG_PATH}")
    source_images = JPEG_PATH

processed_count = {'train': {'benign': 0, 'malignant': 0}, 
                   'test': {'benign': 0, 'malignant': 0}}

# Copier échantillon d'images (limitons pour démo)
MAX_PER_CATEGORY = 50  # Ajustez selon besoin

if path_col:
    for split in ['train', 'test']:
        for label in ['benign', 'malignant']:
            df_subset = df_combined[(df_combined['split'] == split) & 
                                   (df_combined['pathology'] == label)].head(MAX_PER_CATEGORY)
            
            for idx, row in df_subset.iterrows():
                img_path_str = row[path_col]
                
                if pd.isna(img_path_str):
                    continue
                
                img_path = JPEG_PATH / img_path_str
                
                if img_path.exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Améliorer qualité
                        img_resized = cv2.resize(img, (224, 224))
                        img_enhanced = enhance_image_quality(img_resized)
                        
                        # Sauvegarder
                        output_path = FINAL_OUTPUT / 'images' / split / label / f"{split}_{label}_{idx:06d}.jpg"
                        cv2.imwrite(str(output_path), img_enhanced)
                        processed_count[split][label] += 1

print(f"   Images copiées:")
for split in ['train', 'test']:
    for label in ['benign', 'malignant']:
        print(f"      {split}/{label}: {processed_count[split][label]}")

# Sauvegarder CSV consolidé
csv_consolidated = FINAL_OUTPUT / 'dataset_consolidated.csv'
df_combined.to_csv(csv_consolidated, index=False)
print(f"   ✅ CSV consolidé: {csv_consolidated}")

# ==========================================
# 6. RAPPORT FINAL
# ==========================================

final_report = {
    'timestamp': datetime.now().isoformat(),
    'total_samples': len(df_combined),
    'splits': {
        'train': len(df_combined[df_combined['split'] == 'train']),
        'test': len(df_combined[df_combined['split'] == 'test'])
    },
    'labels': df_combined['pathology'].value_counts().to_dict(),
    'images_processed': processed_count,
    'visualizations_created': [
        '09_mosaique_etiquetee_complete.png',
        '10_amelioration_qualite_comparaison.png',
        '11_distribution_complete.png'
    ],
    'quality_enhancements': [
        'CLAHE adaptatif',
        'Débruitage Non-Local Means',
        'Sharpening',
        'Edge enhancement',
        'Normalisation'
    ]
}

report_path = REPORTS_PATH / 'RAPPORT_FINAL_CONSOLIDATION.json'
with open(report_path, 'w') as f:
    json.dump(final_report, f, indent=4, default=str)

print(f"\n💾 Rapport final: {report_path}")

# ==========================================
# 7. RÉSUMÉ
# ==========================================

print(f"\n{'='*80}")
print(f"✨ CONSOLIDATION FINALE TERMINÉE!")
print(f"{'='*80}\n")

print(f"📊 Résumé:")
print(f"   Total échantillons: {final_report['total_samples']:,}")
print(f"   Train: {final_report['splits']['train']:,}")
print(f"   Test: {final_report['splits']['test']:,}")

print(f"\n📁 Fichiers générés:")
print(f"   {FINAL_OUTPUT}/")
print(f"   ├── images/ (organisé par split/label)")
print(f"   └── dataset_consolidated.csv")

print(f"\n🎨 Visualisations:")
for viz in final_report['visualizations_created']:
    print(f"   ✅ {viz}")

print(f"\n🚀 Prochaine étape: Mettre à jour Streamlit")
print(f"{'='*80}\n")
