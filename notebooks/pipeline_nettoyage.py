"""
pipeline_nettoyage.py
PIPELINE DE NETTOYAGE AUTOMATISÉ - Dataset Cancer du Sein
Application des règles médicales avec traçabilité complète

Auteur: TIA Ange Jules-Rihem ben Maouia
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "csv"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned"
LOGS_PATH = BASE_DIR / "data" / "logs"

# Créer les dossiers
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Log global
CLEANING_LOG = {
    'timestamp': datetime.now().isoformat(),
    'operations': []
}

print("="*80)
print("🧹 PIPELINE DE NETTOYAGE AUTOMATISÉ - CANCER DU SEIN")
print("="*80 + "\n")

# ==========================================
# 1. FONCTIONS UTILITAIRES DE LOGGING
# ==========================================

def log_operation(step_name, details, rows_before=None, rows_after=None):
    """Enregistre une opération de nettoyage dans le log"""
    operation = {
        'step': step_name,
        'timestamp': datetime.now().isoformat(),
        'details': details
    }
    
    if rows_before is not None and rows_after is not None:
        operation['rows_before'] = int(rows_before)
        operation['rows_after'] = int(rows_after)
        operation['rows_removed'] = int(rows_before - rows_after)
        operation['percentage_removed'] = f"{((rows_before - rows_after) / rows_before * 100):.2f}%"
    
    CLEANING_LOG['operations'].append(operation)
    print(f"✅ {step_name}: {details}")
    
    if rows_before is not None and rows_after is not None:
        print(f"   Lignes avant: {rows_before:,} | Après: {rows_after:,} | Supprimées: {rows_before - rows_after:,}")

# ==========================================
# 2. DÉTECTION AUTOMATIQUE DE LA COLONNE CIBLE
# ==========================================

def detect_target_column(df):
    """
    Détecte automatiquement la colonne cible (pathology)
    
    Returns:
        str: Nom de la colonne cible ou None
    """
    print("\n📍 ÉTAPE 1: Détection de la colonne cible")
    print("-" * 80)
    
    # Liste de mots-clés possibles
    target_keywords = ['pathology', 'diagnosis', 'label', 'class', 'target']
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                log_operation(
                    "Détection colonne cible",
                    f"Colonne '{col}' identifiée comme variable cible"
                )
                print(f"   ✅ Colonne cible détectée: {col}")
                return col
    
    print("   ⚠️ Aucune colonne cible détectée automatiquement")
    return None

# ==========================================
# 3. NORMALISATION DES VARIABLES CATÉGORIELLES
# ==========================================

def normalize_categorical_variables(df):
    """
    Normalise toutes les variables catégorielles
    - Conversion en minuscules
    - Suppression des espaces
    - Suppression des caractères spéciaux
    """
    print("\n📍 ÉTAPE 2: Normalisation des variables catégorielles")
    print("-" * 80)
    
    df_normalized = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    changes = {}
    
    for col in categorical_cols:
        original_values = df[col].unique()
        
        # Normalisation
        df_normalized[col] = df_normalized[col].astype(str).apply(
            lambda x: x.lower().strip() if x != 'nan' else x
        )
        
        # Supprimer les espaces multiples
        df_normalized[col] = df_normalized[col].apply(
            lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x
        )
        
        normalized_values = df_normalized[col].unique()
        
        if not np.array_equal(original_values, normalized_values):
            changes[col] = {
                'before': len(original_values),
                'after': len(normalized_values),
                'unique_values': normalized_values.tolist()[:10]  # Limiter à 10
            }
    
    if changes:
        log_operation(
            "Normalisation catégorielles",
            f"{len(changes)} colonnes normalisées: {list(changes.keys())}"
        )
        for col, change in changes.items():
            print(f"   {col}: {change['before']} → {change['after']} valeurs uniques")
    else:
        print("   ℹ️ Aucune normalisation nécessaire")
    
    return df_normalized

# ==========================================
# 4. HARMONISATION DE LA VARIABLE PATHOLOGY
# ==========================================

def harmonize_pathology(df, target_col):
    """
    Harmonise les valeurs de la colonne pathology selon les règles médicales
    """
    print("\n📍 ÉTAPE 3: Harmonisation de la variable pathology")
    print("-" * 80)
    
    if target_col is None or target_col not in df.columns:
        print("   ⚠️ Colonne pathology non trouvée, étape ignorée")
        return df
    
    df_harmonized = df.copy()
    
    # Mapping des valeurs
    benign_patterns = ['benign', 'bénin', 'benin', 'negative', 'negatif', 'benign_without_callback']
    malignant_patterns = ['malignant', 'malign', 'malig', 'cancer', 'positive', 'positif']
    
    original_values = df_harmonized[target_col].value_counts()
    
    def harmonize_value(val):
        if pd.isna(val):
            return val
        
        val_str = str(val).lower().strip()
        
        # Vérifier benign
        for pattern in benign_patterns:
            if pattern in val_str:
                return 'benign'
        
        # Vérifier malignant
        for pattern in malignant_patterns:
            if pattern in val_str:
                return 'malignant'
        
        return val  # Conserver si non reconnu
    
    df_harmonized[target_col] = df_harmonized[target_col].apply(harmonize_value)
    
    harmonized_values = df_harmonized[target_col].value_counts()
    
    log_operation(
        "Harmonisation pathology",
        f"Valeurs harmonisées: {harmonized_values.to_dict()}"
    )
    
    print(f"   Distribution avant:")
    for val, count in original_values.items():
        print(f"      {val}: {count}")
    
    print(f"   Distribution après:")
    for val, count in harmonized_values.items():
        print(f"      {val}: {count}")
    
    return df_harmonized

# ==========================================
# 5. GESTION DES DOUBLONS
# ==========================================

def handle_duplicates(df):
    """
    Détecte et supprime les doublons exacts
    """
    print("\n📍 ÉTAPE 4: Gestion des doublons")
    print("-" * 80)
    
    rows_before = len(df)
    
    # Doublons exacts
    n_duplicates = df.duplicated().sum()
    
    if n_duplicates > 0:
        print(f"   ⚠️ {n_duplicates} doublons exacts détectés")
        df_clean = df.drop_duplicates(keep='first')
        rows_after = len(df_clean)
        
        log_operation(
            "Suppression doublons",
            f"{n_duplicates} doublons exacts supprimés",
            rows_before,
            rows_after
        )
    else:
        df_clean = df.copy()
        print(f"   ✅ Aucun doublon détecté")
        log_operation("Vérification doublons", "Aucun doublon trouvé")
    
    return df_clean

# ==========================================
# 6. GESTION DES VALEURS MANQUANTES
# ==========================================

def handle_missing_values(df, target_col, critical_cols=None):
    """
    Gère les valeurs manquantes selon les règles médicales
    
    Args:
        df: DataFrame
        target_col: Colonne cible (pathology)
        critical_cols: Liste des colonnes critiques (si None, détection auto)
    """
    print("\n📍 ÉTAPE 5: Gestion des valeurs manquantes")
    print("-" * 80)
    
    rows_before = len(df)
    df_clean = df.copy()
    
    # Colonnes critiques par défaut
    if critical_cols is None:
        critical_cols = []
        if target_col:
            critical_cols.append(target_col)
        
        # Colonnes BI-RADS et caractéristiques morphologiques
        potential_critical = ['assessment', 'margin', 'margins', 'mass_margins', 
                             'shape', 'mass_shape', 'density', 'breast_density',
                             'calc_type', 'calc_distribution']
        
        for col in potential_critical:
            if col in df.columns:
                critical_cols.append(col)
    
    print(f"   Colonnes critiques identifiées: {critical_cols}")
    
    # Analyse des valeurs manquantes
    missing_stats = []
    for col in df.columns:
        n_missing = df_clean[col].isnull().sum()
        if n_missing > 0:
            pct_missing = (n_missing / len(df_clean)) * 100
            is_critical = col in critical_cols
            
            missing_stats.append({
                'column': col,
                'missing': n_missing,
                'percentage': pct_missing,
                'critical': is_critical
            })
    
    if missing_stats:
        print(f"\n   📊 Valeurs manquantes détectées:")
        for stat in missing_stats:
            status = "🔴 CRITIQUE" if stat['critical'] else "🟡 Non-critique"
            print(f"      {status} {stat['column']}: {stat['missing']} ({stat['percentage']:.2f}%)")
    
    # Supprimer les lignes avec valeurs manquantes dans colonnes critiques
    rows_to_remove = pd.Series([False] * len(df_clean))
    
    for col in critical_cols:
        if col in df_clean.columns:
            missing_mask = df_clean[col].isnull()
            n_missing = missing_mask.sum()
            
            if n_missing > 0:
                rows_to_remove |= missing_mask
                print(f"   ❌ {col}: {n_missing} lignes marquées pour suppression")
    
    total_to_remove = rows_to_remove.sum()
    
    if total_to_remove > 0:
        df_clean = df_clean[~rows_to_remove]
        rows_after = len(df_clean)
        
        log_operation(
            "Suppression valeurs manquantes critiques",
            f"{total_to_remove} lignes avec valeurs critiques manquantes supprimées",
            rows_before,
            rows_after
        )
    else:
        print(f"   ✅ Aucune valeur manquante dans les colonnes critiques")
        log_operation("Vérification valeurs manquantes", "Aucune valeur critique manquante")
    
    return df_clean

# ==========================================
# 7. DÉTECTION D'OUTLIERS
# ==========================================

def detect_outliers(df):
    """
    Détecte les outliers sur les variables quantitatives (âge, subtlety)
    Utilise la méthode IQR
    """
    print("\n📍 ÉTAPE 6: Détection des outliers")
    print("-" * 80)
    
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Colonnes à vérifier
    cols_to_check = []
    for col in ['age', 'subtlety', 'assessment']:
        if col in numeric_cols:
            cols_to_check.append(col)
    
    if not cols_to_check:
        print("   ℹ️ Aucune colonne numérique à vérifier")
        return df_clean
    
    print(f"   Colonnes vérifiées: {cols_to_check}")
    
    outliers_detected = {}
    
    for col in cols_to_check:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            outlier_values = df_clean.loc[outlier_mask, col].values
            outliers_detected[col] = {
                'count': n_outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'values': outlier_values[:10].tolist()  # Limiter à 10
            }
            
            print(f"   ⚠️ {col}: {n_outliers} outliers détectés")
            print(f"      Plage acceptable: [{lower_bound:.1f}, {upper_bound:.1f}]")
            
            # Règles spécifiques pour l'âge
            if col == 'age':
                # Supprimer les âges impossibles
                impossible_age = (df_clean[col] < 0) | (df_clean[col] > 120)
                n_impossible = impossible_age.sum()
                
                if n_impossible > 0:
                    rows_before = len(df_clean)
                    df_clean = df_clean[~impossible_age]
                    rows_after = len(df_clean)
                    
                    log_operation(
                        f"Suppression âges impossibles",
                        f"{n_impossible} âges biologiquement impossibles supprimés",
                        rows_before,
                        rows_after
                    )
            
            # Pour les autres, on annote seulement
            df_clean[f'{col}_outlier'] = outlier_mask
    
    if outliers_detected:
        log_operation(
            "Détection outliers",
            f"Outliers détectés: {outliers_detected}"
        )
    else:
        print(f"   ✅ Aucun outlier détecté")
    
    return df_clean

# ==========================================
# 8. CONTRÔLE COHÉRENCE BI-RADS VS PATHOLOGY
# ==========================================

def check_birads_pathology_consistency(df, target_col):
    """
    Vérifie la cohérence entre assessment (BI-RADS) et pathology
    """
    print("\n📍 ÉTAPE 7: Contrôle de cohérence BI-RADS ↔ Pathology")
    print("-" * 80)
    
    if target_col is None or target_col not in df.columns:
        print("   ⚠️ Colonne pathology non trouvée, vérification impossible")
        return df
    
    if 'assessment' not in df.columns:
        print("   ⚠️ Colonne assessment non trouvée, vérification impossible")
        return df
    
    df_checked = df.copy()
    
    def check_consistency(row):
        """Vérifie la cohérence d'une ligne"""
        birads = row.get('assessment')
        pathology = row.get(target_col)
        
        if pd.isna(birads) or pd.isna(pathology):
            return 'MISSING_DATA'
        
        # Convertir en minuscules pour comparaison
        pathology_str = str(pathology).lower()
        
        try:
            birads_int = int(birads)
        except:
            return 'INVALID_BIRADS'
        
        # Règles de cohérence
        # BI-RADS 1-2 + Malignant = CRITIQUE
        if birads_int in [1, 2] and 'malignant' in pathology_str:
            return 'CRITICAL_INCONSISTENCY'
        
        # BI-RADS 5 + Benign = MODÉRÉ
        if birads_int == 5 and 'benign' in pathology_str:
            return 'MODERATE_INCONSISTENCY'
        
        # BI-RADS 0 = Incomplet
        if birads_int == 0:
            return 'INCOMPLETE_ASSESSMENT'
        
        return 'CONSISTENT'
    
    # Appliquer la vérification
    df_checked['consistency_flag'] = df_checked.apply(check_consistency, axis=1)
    
    # Statistiques de cohérence
    consistency_stats = df_checked['consistency_flag'].value_counts()
    
    print(f"\n   📊 Résultats de cohérence:")
    for flag, count in consistency_stats.items():
        pct = (count / len(df_checked)) * 100
        
        if flag == 'CONSISTENT':
            print(f"      ✅ {flag}: {count} ({pct:.1f}%)")
        elif flag == 'CRITICAL_INCONSISTENCY':
            print(f"      🔴 {flag}: {count} ({pct:.1f}%)")
        elif flag == 'MODERATE_INCONSISTENCY':
            print(f"      🟡 {flag}: {count} ({pct:.1f}%)")
        else:
            print(f"      ⚠️ {flag}: {count} ({pct:.1f}%)")
    
    # Supprimer les incohérences critiques
    critical_mask = df_checked['consistency_flag'] == 'CRITICAL_INCONSISTENCY'
    n_critical = critical_mask.sum()
    
    if n_critical > 0:
        rows_before = len(df_checked)
        df_checked = df_checked[~critical_mask]
        rows_after = len(df_checked)
        
        log_operation(
            "Suppression incohérences critiques",
            f"{n_critical} cas avec incohérence BI-RADS/Pathology critique supprimés",
            rows_before,
            rows_after
        )
    
    log_operation(
        "Vérification cohérence BI-RADS",
        f"Statistiques: {consistency_stats.to_dict()}"
    )
    
    return df_checked

# ==========================================
# 9. PIPELINE COMPLET
# ==========================================

def clean_dataset(df, dataset_name="dataset"):
    """
    Applique le pipeline de nettoyage complet
    
    Args:
        df: DataFrame à nettoyer
        dataset_name: Nom du dataset pour les logs
    
    Returns:
        DataFrame nettoyé
    """
    print(f"\n{'='*80}")
    print(f"🚀 NETTOYAGE DU DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"Dimensions initiales: {len(df):,} lignes × {df.shape[1]} colonnes\n")
    
    # Pipeline séquentiel
    df_cleaned = df.copy()
    
    # 1. Détection colonne cible
    target_col = detect_target_column(df_cleaned)
    
    # 2. Normalisation catégorielles
    df_cleaned = normalize_categorical_variables(df_cleaned)
    
    # 3. Harmonisation pathology
    df_cleaned = harmonize_pathology(df_cleaned, target_col)
    
    # 4. Gestion doublons
    df_cleaned = handle_duplicates(df_cleaned)
    
    # 5. Gestion valeurs manquantes
    df_cleaned = handle_missing_values(df_cleaned, target_col)
    
    # 6. Détection outliers
    df_cleaned = detect_outliers(df_cleaned)
    
    # 7. Cohérence BI-RADS
    df_cleaned = check_birads_pathology_consistency(df_cleaned, target_col)
    
    print(f"\n{'='*80}")
    print(f"✅ NETTOYAGE TERMINÉ: {dataset_name}")
    print(f"{'='*80}")
    print(f"Dimensions finales: {len(df_cleaned):,} lignes × {df_cleaned.shape[1]} colonnes")
    print(f"Réduction: {len(df) - len(df_cleaned):,} lignes ({((len(df) - len(df_cleaned)) / len(df) * 100):.2f}%)")
    
    # Statistiques finales
    if target_col and target_col in df_cleaned.columns:
        print(f"\n📊 Distribution finale de {target_col}:")
        for val, count in df_cleaned[target_col].value_counts().items():
            print(f"   {val}: {count} ({count/len(df_cleaned)*100:.1f}%)")
    
    return df_cleaned

# ==========================================
# 10. EXÉCUTION PRINCIPALE
# ==========================================

if __name__ == "__main__":
    # Liste des fichiers à nettoyer
    files_to_clean = [
        'calc_case_description_train_set.csv',
        'calc_case_description_test_set.csv',
        'mass_case_description_train_set.csv',
        'mass_case_description_test_set.csv'
    ]
    
    cleaned_datasets = {}
    
    for filename in files_to_clean:
        filepath = DATA_PATH / filename
        
        if not filepath.exists():
            print(f"\n⚠️ Fichier non trouvé: {filename}")
            continue
        
        try:
            # Charger le dataset
            print(f"\n📥 Chargement de {filename}...")
            df = pd.read_csv(filepath)
            
            # Nettoyer
            df_cleaned = clean_dataset(df, filename)
            
            # Sauvegarder
            output_filename = filename.replace('.csv', '_cleaned.csv')
            output_path = OUTPUT_PATH / output_filename
            df_cleaned.to_csv(output_path, index=False)
            
            print(f"💾 Dataset nettoyé sauvegardé: {output_path}")
            
            cleaned_datasets[filename] = {
                'rows_before': len(df),
                'rows_after': len(df_cleaned),
                'output_file': str(output_path)
            }
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {filename}: {e}")
            continue
    
    # Sauvegarder le log complet
    CLEANING_LOG['datasets_cleaned'] = cleaned_datasets
    CLEANING_LOG['summary'] = {
        'total_datasets': len(cleaned_datasets),
        'total_operations': len(CLEANING_LOG['operations'])
    }
    
    log_filepath = LOGS_PATH / f"cleaning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filepath, 'w', encoding='utf-8') as f:
        json.dump(CLEANING_LOG, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"📋 RÉSUMÉ FINAL")
    print(f"{'='*80}")
    print(f"Datasets nettoyés: {len(cleaned_datasets)}")
    print(f"Opérations effectuées: {len(CLEANING_LOG['operations'])}")
    print(f"Log sauvegardé: {log_filepath}")
    
    print(f"\n📊 Récapitulatif par dataset:")
    for filename, stats in cleaned_datasets.items():
        reduction = stats['rows_before'] - stats['rows_after']
        pct = (reduction / stats['rows_before'] * 100) if stats['rows_before'] > 0 else 0
        print(f"   {filename}:")
        print(f"      Avant: {stats['rows_before']:,} | Après: {stats['rows_after']:,} | Réduction: {reduction:,} ({pct:.2f}%)")
    
    print(f"\n✨ PIPELINE DE NETTOYAGE TERMINÉ AVEC SUCCÈS!")
    print(f"{'='*80}\n")
