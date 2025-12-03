"""
fusion_data_exploration.py
FUSION ET EXPLORATION COMPLÈTE - CSV + IMAGES
Extrait les caractéristiques et crée un dataset unifié

Fonctionnalités :
1. Extraction features CSV (métadonnées, stats)
2. Extraction features images (histogrammes, textures)
3. Fusion multimodale (CSV + Images)
4. Sauvegarde dans data/features/

Auteur: TIA Ange Jules-Rihem ben Maouia
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "csv"
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned"
JPEG_PATH = BASE_DIR / "jpeg"
PROCESSED_IMAGES_PATH = BASE_DIR / "data" / "processed_images"
FEATURES_PATH = BASE_DIR / "data" / "features"
REPORTS_PATH = BASE_DIR / "reports"

# Créer dossiers
FEATURES_PATH.mkdir(parents=True, exist_ok=True)
(FEATURES_PATH / "csv").mkdir(exist_ok=True)
(FEATURES_PATH / "images").mkdir(exist_ok=True)
(FEATURES_PATH / "fusion").mkdir(exist_ok=True)
(FEATURES_PATH / "visualizations").mkdir(exist_ok=True)

print("="*80)
print("🔬 EXTRACTION DE CARACTÉRISTIQUES ET FUSION CSV + IMAGES")
print("="*80 + "\n")

# ==========================================
# 1. EXTRACTION CARACTÉRISTIQUES CSV
# ==========================================

def extract_csv_features(df, dataset_name):
    """
    Extrait les caractéristiques statistiques d'un dataset CSV
    
    Returns:
        dict: Caractéristiques extraites
    """
    print(f"\n📊 Extraction features CSV: {dataset_name}")
    print("-" * 80)
    
    features = {
        'dataset_name': dataset_name,
        'timestamp': datetime.now().isoformat(),
        
        # Dimensions
        'n_rows': len(df),
        'n_columns': df.shape[1],
        'total_cells': len(df) * df.shape[1],
        
        # Qualité des données
        'missing_cells': int(df.isnull().sum().sum()),
        'missing_percentage': float((df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100),
        'completeness': float(100 - (df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100),
        'duplicates': int(df.duplicated().sum()),
        
        # Par colonne
        'columns': {},
        
        # Distribution pathology
        'pathology_distribution': {},
        
        # Statistiques numériques
        'numeric_stats': {},
        
        # Statistiques catégorielles
        'categorical_stats': {}
    }
    
    # Analyser chaque colonne
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'unique_values': int(df[col].nunique()),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': float((df[col].isnull().sum() / len(df)) * 100)
        }
        
        # Stats numériques
        if df[col].dtype in ['int64', 'float64']:
            col_info['mean'] = float(df[col].mean()) if not df[col].isnull().all() else None
            col_info['std'] = float(df[col].std()) if not df[col].isnull().all() else None
            col_info['min'] = float(df[col].min()) if not df[col].isnull().all() else None
            col_info['max'] = float(df[col].max()) if not df[col].isnull().all() else None
            col_info['median'] = float(df[col].median()) if not df[col].isnull().all() else None
            
            features['numeric_stats'][col] = {
                'mean': col_info['mean'],
                'std': col_info['std'],
                'min': col_info['min'],
                'max': col_info['max']
            }
        
        # Stats catégorielles
        elif df[col].dtype == 'object':
            value_counts = df[col].value_counts().to_dict()
            col_info['top_values'] = {str(k): int(v) for k, v in list(value_counts.items())[:5]}
            
            features['categorical_stats'][col] = col_info['top_values']
        
        features['columns'][col] = col_info
    
    # Distribution pathology
    if 'pathology' in df.columns:
        pathology_dist = df['pathology'].value_counts().to_dict()
        features['pathology_distribution'] = {str(k): int(v) for k, v in pathology_dist.items()}
    
    print(f"   ✅ {features['n_rows']} lignes × {features['n_columns']} colonnes")
    print(f"   ✅ Complétude: {features['completeness']:.2f}%")
    print(f"   ✅ Doublons: {features['duplicates']}")
    
    if features['pathology_distribution']:
        print(f"   ✅ Distribution pathology:")
        for label, count in features['pathology_distribution'].items():
            print(f"      {label}: {count}")
    
    return features

def process_all_csv():
    """Traite tous les CSV et extrait les features"""
    print(f"\n{'='*80}")
    print(f"📋 TRAITEMENT DES CSV")
    print(f"{'='*80}")
    
    all_features = {}
    
    # Chercher les CSV nettoyés d'abord
    csv_files = []
    
    if CLEANED_CSV_PATH.exists():
        csv_files = list(CLEANED_CSV_PATH.glob("*_cleaned.csv"))
        source = "cleaned"
    
    if not csv_files and CSV_PATH.exists():
        csv_files = list(CSV_PATH.glob("*.csv"))
        source = "original"
    
    print(f"   {len(csv_files)} fichiers CSV trouvés ({source})")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            features = extract_csv_features(df, csv_file.name)
            
            # Sauvegarder features individuelles
            output_file = FEATURES_PATH / "csv" / f"{csv_file.stem}_features.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=4, ensure_ascii=False)
            
            all_features[csv_file.name] = features
            
        except Exception as e:
            print(f"   ❌ Erreur sur {csv_file.name}: {e}")
    
    # Sauvegarder résumé global
    summary_file = FEATURES_PATH / "csv_features_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, indent=4, ensure_ascii=False, default=str)
    
    print(f"\n✅ Features CSV sauvegardées dans: {FEATURES_PATH / 'csv'}")
    
    return all_features

# ==========================================
# 2. EXTRACTION CARACTÉRISTIQUES IMAGES
# ==========================================

def extract_image_features(img_path):
    """
    Extrait les caractéristiques d'une image
    
    Returns:
        dict: Features de l'image
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        features = {
            # Dimensions
            'height': int(img.shape[0]),
            'width': int(img.shape[1]),
            'total_pixels': int(img.shape[0] * img.shape[1]),
            
            # Statistiques intensité
            'mean_intensity': float(np.mean(img)),
            'std_intensity': float(np.std(img)),
            'min_intensity': int(np.min(img)),
            'max_intensity': int(np.max(img)),
            'median_intensity': float(np.median(img)),
            
            # Histogramme (quantiles)
            'histogram_quartiles': {
                'q25': float(np.percentile(img, 25)),
                'q50': float(np.percentile(img, 50)),
                'q75': float(np.percentile(img, 75))
            },
            
            # Contraste
            'contrast': float(np.max(img) - np.min(img)),
            'contrast_normalized': float((np.max(img) - np.min(img)) / 255.0),
            
            # Entropie (mesure de complexité)
            'entropy': float(calculate_entropy(img))
        }
        
        return features
        
    except Exception as e:
        return None

def calculate_entropy(img):
    """Calcule l'entropie d'une image (mesure de complexité)"""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normaliser
    hist = hist[hist > 0]  # Enlever les zéros
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def process_images_sample(max_images=100):
    """
    Traite un échantillon d'images et extrait les features
    """
    print(f"\n{'='*80}")
    print(f"🖼️  TRAITEMENT DES IMAGES")
    print(f"{'='*80}")
    
    # Trouver des images
    image_folders = []
    
    # Priorité aux images prétraitées
    if PROCESSED_IMAGES_PATH.exists():
        print(f"   Utilisation des images prétraitées")
        train_benign = list((PROCESSED_IMAGES_PATH / "train" / "benign").glob("*_original.jpg"))
        train_malignant = list((PROCESSED_IMAGES_PATH / "train" / "malignant").glob("*_original.jpg"))
        image_files = train_benign + train_malignant
    else:
        print(f"   Utilisation des images JPEG brutes")
        # Sinon, images brutes
        if JPEG_PATH.exists():
            folders = [f for f in JPEG_PATH.iterdir() if f.is_dir()][:20]
            image_files = []
            for folder in folders:
                imgs = list(folder.glob("*.jpg"))
                if imgs:
                    image_files.append(imgs[0])
        else:
            print("   ⚠️ Aucune image trouvée")
            return None
    
    # Limiter le nombre
    image_files = image_files[:max_images]
    
    print(f"   {len(image_files)} images à analyser")
    
    all_features = []
    
    for img_path in tqdm(image_files, desc="Extraction features"):
        features = extract_image_features(img_path)
        
        if features:
            features['filename'] = img_path.name
            features['path'] = str(img_path)
            all_features.append(features)
    
    # Convertir en DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Statistiques globales
    summary = {
        'n_images': len(df_features),
        'mean_size': {
            'height': float(df_features['height'].mean()),
            'width': float(df_features['width'].mean())
        },
        'intensity_stats': {
            'mean': float(df_features['mean_intensity'].mean()),
            'std': float(df_features['std_intensity'].mean()),
            'min': float(df_features['min_intensity'].min()),
            'max': float(df_features['max_intensity'].max())
        },
        'contrast_stats': {
            'mean': float(df_features['contrast'].mean()),
            'std': float(df_features['contrast'].std())
        },
        'entropy_stats': {
            'mean': float(df_features['entropy'].mean()),
            'std': float(df_features['entropy'].std())
        }
    }
    
    # Sauvegarder features individuelles
    df_features.to_csv(FEATURES_PATH / "images" / "image_features.csv", index=False)
    
    # Sauvegarder résumé
    with open(FEATURES_PATH / "images" / "image_features_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✅ Features images sauvegardées:")
    print(f"   Moyenne intensité: {summary['intensity_stats']['mean']:.2f}")
    print(f"   Contraste moyen: {summary['contrast_stats']['mean']:.2f}")
    print(f"   Entropie moyenne: {summary['entropy_stats']['mean']:.2f}")
    
    return df_features

# ==========================================
# 3. FUSION CSV + IMAGES
# ==========================================

def create_multimodal_fusion():
    """
    Crée un dataset fusionné CSV + Images
    """
    print(f"\n{'='*80}")
    print(f"🔗 FUSION MULTIMODALE CSV + IMAGES")
    print(f"{'='*80}")
    
    # Charger les features CSV
    csv_features_file = FEATURES_PATH / "csv_features_summary.json"
    if not csv_features_file.exists():
        print("   ⚠️ Features CSV non trouvées, exécutez extract_csv_features d'abord")
        return None
    
    with open(csv_features_file, 'r') as f:
        csv_features = json.load(f)
    
    # Charger les features images
    img_features_file = FEATURES_PATH / "images" / "image_features.csv"
    if not img_features_file.exists():
        print("   ⚠️ Features images non trouvées, exécutez extract_image_features d'abord")
        return None
    
    df_img_features = pd.read_csv(img_features_file)
    
    # Créer un résumé de fusion
    fusion_summary = {
        'timestamp': datetime.now().isoformat(),
        'csv_datasets': list(csv_features.keys()),
        'n_images': len(df_img_features),
        
        'csv_summary': {
            'total_rows': sum(f['n_rows'] for f in csv_features.values()),
            'total_columns': sum(f['n_columns'] for f in csv_features.values()) // len(csv_features),
            'avg_completeness': np.mean([f['completeness'] for f in csv_features.values()])
        },
        
        'image_summary': {
            'total_images': len(df_img_features),
            'mean_intensity': float(df_img_features['mean_intensity'].mean()),
            'mean_contrast': float(df_img_features['contrast'].mean()),
            'mean_entropy': float(df_img_features['entropy'].mean())
        },
        
        'fusion_potential': {
            'description': 'Dataset multimodal combinant métadonnées CSV et features images',
            'applications': [
                'Entraînement modèles multimodaux',
                'Prédiction combinant données tabulaires + images',
                'Analyse de cohérence CSV <-> Images'
            ]
        }
    }
    
    # Sauvegarder
    fusion_file = FEATURES_PATH / "fusion" / "multimodal_fusion_summary.json"
    with open(fusion_file, 'w', encoding='utf-8') as f:
        json.dump(fusion_summary, f, indent=4, ensure_ascii=False)
    
    # Créer un fichier CSV combiné (exemple)
    # On ajoute des stats d'images au CSV
    sample_fusion = pd.DataFrame({
        'source': ['CSV'] * 3 + ['Images'] * 3,
        'metric': [
            'Total lignes',
            'Complétude (%)',
            'Doublons'
        ] + [
            'Total images',
            'Intensité moyenne',
            'Contraste moyen'
        ],
        'value': [
            fusion_summary['csv_summary']['total_rows'],
            fusion_summary['csv_summary']['avg_completeness'],
            sum(f.get('duplicates', 0) for f in csv_features.values())
        ] + [
            fusion_summary['image_summary']['total_images'],
            fusion_summary['image_summary']['mean_intensity'],
            fusion_summary['image_summary']['mean_contrast']
        ]
    })
    
    sample_fusion.to_csv(FEATURES_PATH / "fusion" / "fusion_metrics.csv", index=False)
    
    print(f"\n✅ Fusion créée:")
    print(f"   CSV: {fusion_summary['csv_summary']['total_rows']} lignes")
    print(f"   Images: {fusion_summary['image_summary']['total_images']} images")
    print(f"   Complétude CSV: {fusion_summary['csv_summary']['avg_completeness']:.2f}%")
    
    return fusion_summary

# ==========================================
# 4. VISUALISATIONS
# ==========================================

def create_visualizations(csv_features, img_features_df, fusion_summary):
    """Crée des visualisations des features"""
    print(f"\n📊 Création des visualisations...")
    
    # Figure 1: Qualité des CSV
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Complétude par dataset
    if csv_features:
        datasets = list(csv_features.keys())
        completeness = [f['completeness'] for f in csv_features.values()]
        
        axes[0].barh(datasets, completeness, color='lightblue')
        axes[0].set_xlabel('Complétude (%)')
        axes[0].set_title('Qualité des Données CSV', fontweight='bold')
        axes[0].axvline(x=95, color='green', linestyle='--', label='Seuil 95%')
        axes[0].legend()
    
    # Distribution intensité images
    if img_features_df is not None and 'mean_intensity' in img_features_df.columns:
        axes[1].hist(img_features_df['mean_intensity'], bins=30, color='coral', edgecolor='black')
        axes[1].set_xlabel('Intensité Moyenne')
        axes[1].set_ylabel('Nombre d\'images')
        axes[1].set_title('Distribution Intensité Images', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FEATURES_PATH / "visualizations" / "features_overview.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ features_overview.png")
    plt.close()
    
    # Figure 2: Fusion metrics
    if fusion_summary:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Lignes CSV', 'Images', 'Complétude CSV (%)', 'Intensité moy.']
        values = [
            fusion_summary['csv_summary']['total_rows'],
            fusion_summary['image_summary']['total_images'],
            fusion_summary['csv_summary']['avg_completeness'],
            fusion_summary['image_summary']['mean_intensity']
        ]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Valeur')
        ax.set_title('Métriques de Fusion CSV + Images', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Annotations
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(FEATURES_PATH / "visualizations" / "fusion_metrics.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ fusion_metrics.png")
        plt.close()

# ==========================================
# 5. EXÉCUTION PRINCIPALE
# ==========================================

if __name__ == "__main__":
    print(f"Dossier de sortie: {FEATURES_PATH}\n")
    
    # 1. Extraire features CSV
    csv_features = process_all_csv()
    
    # 2. Extraire features images
    img_features_df = process_images_sample(max_images=100)
    
    # 3. Créer fusion
    fusion_summary = create_multimodal_fusion()
    
    # 4. Visualisations
    if csv_features or img_features_df is not None:
        create_visualizations(csv_features, img_features_df, fusion_summary)
    
    # Rapport final
    print(f"\n{'='*80}")
    print(f"📋 RAPPORT FINAL")
    print(f"{'='*80}\n")
    
    print(f"✅ Features extraites et sauvegardées:")
    print(f"   📂 {FEATURES_PATH}/")
    print(f"   ├── csv/")
    print(f"   │   └── *_features.json")
    print(f"   ├── images/")
    print(f"   │   ├── image_features.csv")
    print(f"   │   └── image_features_summary.json")
    print(f"   ├── fusion/")
    print(f"   │   ├── multimodal_fusion_summary.json")
    print(f"   │   └── fusion_metrics.csv")
    print(f"   └── visualizations/")
    print(f"       ├── features_overview.png")
    print(f"       └── fusion_metrics.png")
    
    print(f"\n✨ EXTRACTION ET FUSION TERMINÉES AVEC SUCCÈS!")
    print(f"{'='*80}\n")
