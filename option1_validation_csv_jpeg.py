"""
OPTION 1 : VALIDATION CSV ↔ JPEG
Vérifie la cohérence entre les fichiers CSV et les images JPEG

Fonctionnalités :
- Vérifie que chaque ligne CSV a son image correspondante
- Identifie les images orphelines (sans CSV)
- Identifie les CSV sans images
- Génère rapport détaillé

Auteur: TIA Ange Jules-Rihem ben Maouia
Date: Décembre 2025
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "csv"
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned" / "csv"
JPEG_PATH = BASE_DIR / "jpeg"
REPORTS_PATH = BASE_DIR / "reports"

print("="*80)
print("🔍 VALIDATION COHÉRENCE CSV ↔ JPEG")
print("="*80 + "\n")

# Utiliser CSV nettoyés si disponibles
csv_source = CLEANED_CSV_PATH if CLEANED_CSV_PATH.exists() and list(CLEANED_CSV_PATH.glob("*.csv")) else CSV_PATH

print(f"Source CSV: {csv_source}")
print(f"Source JPEG: {JPEG_PATH}\n")

validation_report = {
    'timestamp': datetime.now().isoformat(),
    'datasets': {},
    'summary': {
        'total_csv_rows': 0,
        'images_found': 0,
        'images_missing': 0,
        'orphan_images': 0
    },
    'missing_images': [],
    'orphan_folders': []
}

# ==========================================
# 1. VALIDATION PAR DATASET CSV
# ==========================================

print("📊 Validation des datasets CSV")
print("-" * 80)

csv_files = list(csv_source.glob("*.csv"))

for csv_file in csv_files:
    print(f"\n📄 {csv_file.name}")
    
    df = pd.read_csv(csv_file)
    validation_report['summary']['total_csv_rows'] += len(df)
    
    dataset_report = {
        'total_rows': len(df),
        'images_found': 0,
        'images_missing': 0,
        'missing_paths': []
    }
    
    # Colonnes possibles pour le chemin
    path_cols = ['image file path', 'image_file_path', 'cropped image file path']
    path_col = None
    
    for col in path_cols:
        if col in df.columns:
            path_col = col
            break
    
    if not path_col:
        print(f"   ⚠️ Aucune colonne de chemin d'image trouvée")
        validation_report['datasets'][csv_file.name] = dataset_report
        continue
    
    print(f"   Colonne utilisée: '{path_col}'")
    
    # Vérifier chaque image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Vérification"):
        img_path_str = row[path_col]
        
        if pd.isna(img_path_str):
            dataset_report['images_missing'] += 1
            continue
        
        # Construire le chemin complet
        img_path = JPEG_PATH / img_path_str
        
        if img_path.exists():
            dataset_report['images_found'] += 1
            validation_report['summary']['images_found'] += 1
        else:
            dataset_report['images_missing'] += 1
            dataset_report['missing_paths'].append(str(img_path_str))
            validation_report['summary']['images_missing'] += 1
            validation_report['missing_images'].append({
                'csv': csv_file.name,
                'row': idx,
                'path': str(img_path_str)
            })
    
    # Afficher résultats
    print(f"   ✅ Trouvées: {dataset_report['images_found']}")
    print(f"   ❌ Manquantes: {dataset_report['images_missing']}")
    print(f"   📊 Taux de couverture: {(dataset_report['images_found']/len(df)*100):.2f}%")
    
    validation_report['datasets'][csv_file.name] = dataset_report

# ==========================================
# 2. IDENTIFICATION IMAGES ORPHELINES
# ==========================================

print(f"\n\n🔍 Identification des images orphelines")
print("-" * 80)

if JPEG_PATH.exists():
    all_jpeg_folders = list(JPEG_PATH.glob("*"))
    print(f"Total dossiers JPEG: {len(all_jpeg_folders)}")
    
    # Collecter tous les chemins référencés dans les CSV
    referenced_paths = set()
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        for col in ['image file path', 'image_file_path', 'cropped image file path']:
            if col in df.columns:
                paths = df[col].dropna()
                # Extraire le nom du dossier (premier niveau du chemin)
                folder_names = paths.apply(lambda x: str(x).split('/')[0] if '/' in str(x) else str(x).split('\\')[0])
                referenced_paths.update(folder_names)
                break
    
    print(f"Dossiers référencés dans CSV: {len(referenced_paths)}")
    
    # Identifier orphelins
    orphan_count = 0
    for folder in all_jpeg_folders:
        if folder.is_dir() and folder.name not in referenced_paths:
            orphan_count += 1
            if orphan_count <= 100:  # Limiter la liste
                validation_report['orphan_folders'].append(folder.name)
    
    validation_report['summary']['orphan_images'] = orphan_count
    
    print(f"   ⚠️ Dossiers orphelins: {orphan_count}")
    print(f"   📊 Taux d'utilisation: {((len(all_jpeg_folders)-orphan_count)/len(all_jpeg_folders)*100):.2f}%")

# ==========================================
# 3. RÉSUMÉ ET SAUVEGARDE
# ==========================================

print(f"\n{'='*80}")
print(f"📋 RÉSUMÉ DE LA VALIDATION")
print(f"{'='*80}\n")

print(f"📊 Statistiques globales:")
print(f"   Total lignes CSV: {validation_report['summary']['total_csv_rows']:,}")
print(f"   Images trouvées: {validation_report['summary']['images_found']:,}")
print(f"   Images manquantes: {validation_report['summary']['images_missing']:,}")
print(f"   Dossiers orphelins: {validation_report['summary']['orphan_images']:,}")

if validation_report['summary']['total_csv_rows'] > 0:
    coverage = (validation_report['summary']['images_found'] / validation_report['summary']['total_csv_rows']) * 100
    print(f"\n✅ Taux de couverture global: {coverage:.2f}%")

# Sauvegarder rapport
report_path = REPORTS_PATH / 'validation_csv_jpeg.json'
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(validation_report, f, indent=4, ensure_ascii=False)

print(f"\n💾 Rapport sauvegardé: {report_path}")

# Créer rapport markdown
md_report = f"""# RAPPORT DE VALIDATION CSV ↔ JPEG

**Date** : {datetime.now().strftime('%d/%m/%Y %H:%M')}

## Résumé Exécutif

- **Total lignes CSV** : {validation_report['summary']['total_csv_rows']:,}
- **Images trouvées** : {validation_report['summary']['images_found']:,}
- **Images manquantes** : {validation_report['summary']['images_missing']:,}
- **Dossiers orphelins** : {validation_report['summary']['orphan_images']:,}
- **Taux de couverture** : {coverage:.2f}%

## Détails par Dataset

"""

for dataset_name, data in validation_report['datasets'].items():
    md_report += f"""
### {dataset_name}

- Lignes: {data['total_rows']:,}
- Images trouvées: {data['images_found']:,}
- Images manquantes: {data['images_missing']:,}
- Couverture: {(data['images_found']/data['total_rows']*100):.2f}%

"""

md_report += f"""
## Recommandations

"""

if validation_report['summary']['images_missing'] > 0:
    md_report += f"- ⚠️ {validation_report['summary']['images_missing']} images manquantes à investiguer\n"

if validation_report['summary']['orphan_images'] > 0:
    md_report += f"- 📁 {validation_report['summary']['orphan_images']} dossiers orphelins (non référencés dans CSV)\n"

if coverage > 95:
    md_report += "- ✅ Excellente couverture (>95%), dataset exploitable\n"

md_report_path = REPORTS_PATH / 'validation_csv_jpeg.md'
with open(md_report_path, 'w', encoding='utf-8') as f:
    f.write(md_report)

print(f"📄 Rapport markdown: {md_report_path}")

print(f"\n✨ VALIDATION TERMINÉE!")
print(f"{'='*80}\n")
