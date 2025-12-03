"""
OPTION 3 : DATASET MULTIMODAL FUSIONNÉ
Crée un dataset ML-ready combinant features CSV + Images

Fonctionnalités :
- Extrait features de tous les CSV
- Extrait features de toutes les images
- Fusionne en dataset unique
- Export CSV, JSON, et numpy arrays
- Prêt pour modélisation

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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "csv"
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned" / "csv"
JPEG_PATH = BASE_DIR / "jpeg"
OUTPUT_PATH = BASE_DIR / "data" / "multimodal_dataset"
REPORTS_PATH = BASE_DIR / "reports"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🔗 CRÉATION DATASET MULTIMODAL FUSIONNÉ")
print("="*80 + "\n")

# ==========================================
# 1. EXTRACTION FEATURES CSV
# ==========================================

print("📊 Extraction features CSV")
print("-" * 80)

csv_source = CLEANED_CSV_PATH if CLEANED_CSV_PATH.exists() and list(CLEANED_CSV_PATH.glob("*.csv")) else CSV_PATH

all_csv_data = []
csv_features_list = []

csv_files = list(csv_source.glob("*.csv"))

for csv_file in csv_files:
    print(f"\n📄 {csv_file.name}")
    
    df = pd.read_csv(csv_file)
    
    # Ajouter split info
    df['split'] = 'train' if 'train' in csv_file.name.lower() else 'test'
    df['dataset'] = csv_file.stem
    
    # Harmoniser pathology
    if 'pathology' in df.columns:
        df['pathology'] = df['pathology'].astype(str).str.lower().str.strip()
        df['pathology'] = df['pathology'].replace({'benign_without_callback': 'benign'})
    
    all_csv_data.append(df)
    print(f"   {len(df)} lignes ajoutées")

# Combiner tous les CSV
df_combined = pd.concat(all_csv_data, ignore_index=True)
print(f"\n✅ Total dataset CSV: {len(df_combined)} lignes\n")

# Sélectionner features numériques et catégorielles
numeric_features = df_combined.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_combined.select_dtypes(include=['object']).columns.tolist()

# Retirer colonnes non-features
exclude_cols = ['patient_id', 'image file path', 'cropped image file path', 'ROI mask file path', 
                'image_file_path', 'dataset']
numeric_features = [f for f in numeric_features if f not in exclude_cols]
categorical_features = [f for f in categorical_features if f not in exclude_cols and f != 'pathology']

print(f"Features numériques ({len(numeric_features)}): {numeric_features[:5]}...")
print(f"Features catégorielles ({len(categorical_features)}): {categorical_features[:3]}...")

# ==========================================
# 2. EXTRACTION FEATURES IMAGES
# ==========================================

print(f"\n\n🖼️  Extraction features images")
print("-" * 80)

def extract_image_features(img_path):
    """Extrait features d'une image"""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Redimensionner pour cohérence
        img = cv2.resize(img, (224, 224))
        
        features = {
            'img_mean': float(np.mean(img)),
            'img_std': float(np.std(img)),
            'img_min': int(np.min(img)),
            'img_max': int(np.max(img)),
            'img_median': float(np.median(img)),
            'img_q25': float(np.percentile(img, 25)),
            'img_q75': float(np.percentile(img, 75)),
            'img_contrast': float(np.max(img) - np.min(img)),
            'img_range_normalized': float((np.max(img) - np.min(img)) / 255.0)
        }
        
        # Histogramme features (3 bins)
        hist = cv2.calcHist([img], [0], None, [3], [0, 256])
        hist = hist.flatten() / hist.sum()
        for i, val in enumerate(hist):
            features[f'img_hist_bin{i}'] = float(val)
        
        return features
    
    except:
        return None

# Trouver colonne chemin
path_col = 'image file path'
if path_col not in df_combined.columns:
    path_col = 'image_file_path'
if path_col not in df_combined.columns:
    path_col = 'cropped image file path'

if path_col in df_combined.columns:
    print(f"Extraction des features images (peut prendre plusieurs minutes)...\n")
    
    # Limiter pour démo (retirer cette ligne pour traiter TOUT)
    SAMPLE_SIZE = 1000  # Traiter 1000 images (retirez cette ligne pour tout traiter)
    df_sample = df_combined.head(SAMPLE_SIZE).copy()
    
    image_features_list = []
    valid_indices = []
    
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Extraction"):
        img_path_str = row[path_col]
        
        if pd.isna(img_path_str):
            continue
        
        img_path = JPEG_PATH / img_path_str
        
        if not img_path.exists():
            continue
        
        img_features = extract_image_features(img_path)
        
        if img_features:
            image_features_list.append(img_features)
            valid_indices.append(idx)
    
    print(f"\n✅ Features extraites pour {len(image_features_list)} images")
    
    # Créer DataFrame features images
    df_image_features = pd.DataFrame(image_features_list, index=valid_indices)
    
    # Fusionner avec CSV
    df_fused = df_sample.loc[valid_indices].copy()
    df_fused = pd.concat([df_fused.reset_index(drop=True), df_image_features.reset_index(drop=True)], axis=1)
    
else:
    print("⚠️ Colonne de chemin d'image non trouvée")
    df_fused = df_combined.copy()
    df_image_features = pd.DataFrame()

# ==========================================
# 3. PRÉPARATION POUR ML
# ==========================================

print(f"\n\n🤖 Préparation pour Machine Learning")
print("-" * 80)

# Encoder les labels
if 'pathology' in df_fused.columns:
    label_encoder = LabelEncoder()
    df_fused['label_encoded'] = label_encoder.fit_transform(df_fused['pathology'])
    
    print(f"Labels encodés:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"   {label} → {idx}")
    
    # Sauvegarder mapping
    label_mapping = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
else:
    label_mapping = {}

# Features finales pour ML
ml_features = numeric_features.copy()

# Ajouter features images si disponibles
if not df_image_features.empty:
    img_feature_cols = df_image_features.columns.tolist()
    ml_features.extend(img_feature_cols)
    print(f"\nFeatures totales: {len(ml_features)}")
    print(f"   - CSV: {len(numeric_features)}")
    print(f"   - Images: {len(img_feature_cols)}")

# Créer matrices X, y
available_features = [f for f in ml_features if f in df_fused.columns]
X = df_fused[available_features].fillna(0).values
y = df_fused['label_encoded'].values if 'label_encoded' in df_fused.columns else None

print(f"\nShape finale:")
print(f"   X: {X.shape}")
if y is not None:
    print(f"   y: {y.shape}")

# Normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. EXPORT
# ==========================================

print(f"\n\n💾 Export du dataset")
print("-" * 80)

# Export CSV complet
csv_export_path = OUTPUT_PATH / 'multimodal_dataset_full.csv'
df_fused.to_csv(csv_export_path, index=False)
print(f"✅ CSV: {csv_export_path}")

# Export features + labels (numpy)
np.save(OUTPUT_PATH / 'X_features.npy', X)
np.save(OUTPUT_PATH / 'X_features_scaled.npy', X_scaled)
if y is not None:
    np.save(OUTPUT_PATH / 'y_labels.npy', y)
print(f"✅ Numpy arrays: X_features.npy, X_features_scaled.npy, y_labels.npy")

# Export train/test séparés
if 'split' in df_fused.columns:
    train_mask = df_fused['split'] == 'train'
    test_mask = df_fused['split'] == 'test'
    
    df_fused[train_mask].to_csv(OUTPUT_PATH / 'train_multimodal.csv', index=False)
    df_fused[test_mask].to_csv(OUTPUT_PATH / 'test_multimodal.csv', index=False)
    
    np.save(OUTPUT_PATH / 'X_train.npy', X[train_mask])
    np.save(OUTPUT_PATH / 'X_test.npy', X[test_mask])
    if y is not None:
        np.save(OUTPUT_PATH / 'y_train.npy', y[train_mask])
        np.save(OUTPUT_PATH / 'y_test.npy', y[test_mask])
    
    print(f"✅ Train/Test splits:  train_multimodal.csv, test_multimodal.csv")

# Métadonnées
metadata = {
    'timestamp': datetime.now().isoformat(),
    'total_samples': len(df_fused),
    'n_features': len(available_features),
    'features': {
        'numeric_csv': numeric_features,
        'image_features': img_feature_cols if not df_image_features.empty else [],
        'total': available_features
    },
    'label_mapping': label_mapping,
    'shape': {
        'X': list(X.shape),
        'y': list(y.shape) if y is not None else None
    },
    'splits': {
        'train': int(train_mask.sum()) if 'split' in df_fused.columns else None,
        'test': int(test_mask.sum()) if 'split' in df_fused.columns else None
    }
}

with open(OUTPUT_PATH / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"✅ Metadata: metadata.json")

# ==========================================
# 5. RÉSUMÉ
# ==========================================

print(f"\n{'='*80}")
print(f"📋 RÉSUMÉ DATASET MULTIMODAL")
print(f"{'='*80}\n")

print(f"✅ Dataset créé avec:")
print(f"   Total échantillons: {len(df_fused):,}")
print(f"   Features totales: {len(available_features)}")
print(f"   Labels: {list(label_mapping.keys())}")

if 'split' in df_fused.columns:
    print(f"\n   Train: {train_mask.sum():,}")
    print(f"   Test: {test_mask.sum():,}")

print(f"\n📂 Fichiers générés dans: {OUTPUT_PATH}/")
print(f"   - multimodal_dataset_full.csv")
print(f"   - X_features*.npy")
print(f"   - y_labels.npy")
print(f"   - train/test splits")
print(f"   - metadata.json")

print(f"\n✨ DATASET MULTIMODAL PRÊT POUR ML!")
print(f"{'='*80}\n")
