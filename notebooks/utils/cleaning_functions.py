"""
Fonctions Utilitaires pour le Nettoyage des Données
Mini-Projet : Évaluation et Amélioration de la Qualité des Données
Dataset : Cancer du Sein (Mammographie)

Auteur: TIA Ange Jules-Rihem ben Maouia
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ===========================
# 1. FONCTIONS D'ANALYSE
# ===========================

def analyse_qualite_globale(df, nom_dataset="Dataset"):
    """
    Analyse globale de la qualité d'un DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame à analyser
    nom_dataset : str
        Nom du dataset pour l'affichage
    
    Returns:
    --------
    dict : Dictionnaire contenant les métriques de qualité
    """
    print(f"\n{'='*60}")
    print(f"ANALYSE DE QUALITÉ : {nom_dataset}")
    print(f"{'='*60}\n")
    
    # Dimensions
    n_lignes, n_colonnes = df.shape
    print(f"📊 Dimensions: {n_lignes:,} lignes × {n_colonnes} colonnes")
    
    # Mémoire
    memoire_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"💾 Mémoire utilisée: {memoire_mb:.2f} MB")
    
    # Valeurs manquantes
    total_cells = n_lignes * n_colonnes
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    print(f"\n❓ Valeurs manquantes: {missing_cells:,} / {total_cells:,} ({missing_pct:.2f}%)")
    
    # Doublons
    n_duplicates = df.duplicated().sum()
    dup_pct = (n_duplicates / n_lignes) * 100
    print(f"🔄 Doublons (lignes complètes): {n_duplicates:,} ({dup_pct:.2f}%)")
    
    # Types de données
    print(f"\n📋 Types de données:")
    for dtype in df.dtypes.value_counts().items():
        print(f"   - {dtype[0]}: {dtype[1]} colonnes")
    
    # Métriques compilées
    metrics = {
        'n_lignes': n_lignes,
        'n_colonnes': n_colonnes,
        'memoire_mb': memoire_mb,
        'missing_cells': missing_cells,
        'missing_pct': missing_pct,
        'n_duplicates': n_duplicates,
        'dup_pct': dup_pct
    }
    
    return metrics


def analyse_valeurs_manquantes(df, seuil_affichage=0):
    """
    Analyse détaillée des valeurs manquantes par colonne
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame à analyser
    seuil_affichage : float
        Seuil minimal (%) pour afficher une colonne (défaut: 0)
    
    Returns:
    --------
    pandas.DataFrame : Rapport des valeurs manquantes
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Colonne': df.columns,
        'Valeurs_Manquantes': missing.values,
        'Pourcentage': missing_pct.values,
        'Type': df.dtypes.values
    })
    
    missing_df = missing_df[missing_df['Pourcentage'] > seuil_affichage]
    missing_df = missing_df.sort_values('Pourcentage', ascending=False)
    
    return missing_df.reset_index(drop=True)


def detecter_doublons_avances(df, colonnes=None, similarite_seuil=90):
    """
    Détecte les doublons exacts et quasi-doublons
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame à analyser
    colonnes : list, optional
        Liste des colonnes à considérer (None = toutes)
    similarite_seuil : int
        Seuil de similarité pour les quasi-doublons (0-100)
    
    Returns:
    --------
    tuple : (doublons_exacts, indices_doublons)
    """
    if colonnes is None:
        subset_df = df
    else:
        subset_df = df[colonnes]
    
    # Doublons exacts
    doublons_exacts = df[subset_df.duplicated(keep=False)]
    
    # Statistiques
    n_doublons = subset_df.duplicated().sum()
    n_unique = len(subset_df.drop_duplicates())
    
    print(f"🔍 Doublons Exacts:")
    print(f"   - Lignes en doublon: {n_doublons:,}")
    print(f"   - Lignes uniques: {n_unique:,}")
    print(f"   - Total: {len(df):,}")
    
    return doublons_exacts, subset_df[subset_df.duplicated()].index.tolist()


def analyser_outliers_numeriques(df, colonnes=None, methode='iqr', seuil=1.5):
    """
    Détecte les outliers dans les colonnes numériques
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame à analyser
    colonnes : list, optional
        Liste des colonnes numériques (None = toutes)
    methode : str
        'iqr' ou 'zscore'
    seuil : float
        Multiplicateur IQR ou seuil z-score
    
    Returns:
    --------
    dict : Dictionnaire {colonne: indices_outliers}
    """
    if colonnes is None:
        colonnes = df.select_dtypes(include=[np.number]).columns
    
    outliers_dict = {}
    
    for col in colonnes:
        if methode == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - seuil * IQR
            upper = Q3 + seuil * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)].index
        elif methode == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > seuil].index
        
        if len(outliers) > 0:
            outliers_dict[col] = outliers.tolist()
            print(f"   {col}: {len(outliers)} outliers")
    
    return outliers_dict


# ===========================
# 2. FONCTIONS DE NETTOYAGE
# ===========================

def supprimer_doublons(df, colonnes=None, garder='first', inplace=False):
    """
    Supprime les lignes en double
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame à nettoyer
    colonnes : list, optional
        Colonnes à considérer pour les doublons
    garder : str
        'first', 'last' ou False (supprimer tous)
    inplace : bool
        Modifier le DataFrame directement
    
    Returns:
    --------
    pandas.DataFrame (si inplace=False)
    int : Nombre de doublons supprimés
    """
    n_avant = len(df)
    df_clean = df.drop_duplicates(subset=colonnes, keep=garder, inplace=inplace)
    n_apres = len(df) if inplace else len(df_clean)
    n_supprimes = n_avant - n_apres
    
    print(f"✅ Doublons supprimés: {n_supprimes:,}")
    
    return df_clean if not inplace else n_supprimes


def imputer_valeurs_manquantes(df, strategie='auto', colonnes=None):
    """
    Impute les valeurs manquantes selon une stratégie
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame à nettoyer
    strategie : str ou dict
        'auto', 'median', 'mean', 'mode', 'ffill', 'bfill'
        ou dict {colonne: strategie}
    colonnes : list, optional
        Colonnes à traiter (None = toutes avec missing)
    
    Returns:
    --------
    pandas.DataFrame : DataFrame avec valeurs imputées
    dict : Log des imputations
    """
    df_clean = df.copy()
    log_imputation = {}
    
    if colonnes is None:
        colonnes = df.columns[df.isnull().any()].tolist()
    
    for col in colonnes:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        
        # Déterminer la stratégie
        if isinstance(strategie, dict):
            strat = strategie.get(col, 'auto')
        else:
            strat = strategie
        
        # Auto : choisir selon le type
        if strat == 'auto':
            if df[col].dtype in ['object', 'category']:
                strat = 'mode'
            else:
                strat = 'median'
        
        # Appliquer l'imputation
        if strat == 'median':
            valeur = df[col].median()
            df_clean[col].fillna(valeur, inplace=True)
        elif strat == 'mean':
            valeur = df[col].mean()
            df_clean[col].fillna(valeur, inplace=True)
        elif strat == 'mode':
            valeur = df[col].mode()[0] if not df[col].mode().empty else None
            df_clean[col].fillna(valeur, inplace=True)
        elif strat == 'ffill':
            df_clean[col].fillna(method='ffill', inplace=True)
        elif strat == 'bfill':
            df_clean[col].fillna(method='bfill', inplace=True)
        
        log_imputation[col] = {
            'n_imputed': n_missing,
            'strategie': strat,
            'valeur': valeur if strat in ['median', 'mean', 'mode'] else None
        }
        
        print(f"   {col}: {n_missing} valeurs imputées ({strat})")
    
    return df_clean, log_imputation


def normaliser_texte(serie, lowercase=True, strip=True, remove_special=False):
    """
    Normalise une série de texte
    
    Parameters:
    -----------
    serie : pandas.Series
        Série de texte à normaliser
    lowercase : bool
        Convertir en minuscules
    strip : bool
        Supprimer espaces début/fin
    remove_special : bool
        Supprimer caractères spéciaux
    
    Returns:
    --------
    pandas.Series : Série normalisée
    """
    serie_clean = serie.copy()
    
    if lowercase:
        serie_clean = serie_clean.str.lower()
    
    if strip:
        serie_clean = serie_clean.str.strip()
    
    if remove_special:
        import re
        serie_clean = serie_clean.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
    
    return serie_clean


# ===========================
# 3. FONCTIONS DE VISUALISATION
# ===========================

def plot_valeurs_manquantes(df, figsize=(12, 6)):
    """
    Visualise les valeurs manquantes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Heatmap
    missing = df.isnull()
    axes[0].imshow(missing.T, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    axes[0].set_title('Carte des Valeurs Manquantes')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Colonnes')
    
    # Bar chart
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    missing_pct.plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Pourcentage de Valeurs Manquantes')
    axes[1].set_xlabel('Pourcentage (%)')
    
    plt.tight_layout()
    return fig


def plot_avant_apres(df_avant, df_apres, metrique='taille'):
    """
    Compare avant/après nettoyage
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if metrique == 'taille':
        data = {
            'Avant': [len(df_avant), df_avant.shape[1]],
            'Après': [len(df_apres), df_apres.shape[1]]
        }
        df_comp = pd.DataFrame(data, index=['Lignes', 'Colonnes'])
        df_comp.plot(kind='bar', ax=ax, color=['#e74c3c', '#27ae60'])
        ax.set_title('Comparaison Taille Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Nombre')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


# ===========================
# 4. FONCTIONS DE LOGGING
# ===========================

def creer_log_nettoyage(nom_operation, details, timestamp=None):
    """
    Crée une entrée de log pour une opération de nettoyage
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        'timestamp': timestamp,
        'operation': nom_operation,
        'details': details
    }
    
    return log_entry


def sauvegarder_log(logs, filepath='data/logs/cleaning_log.csv'):
    """
    Sauvegarde les logs dans un fichier CSV
    """
    df_log = pd.DataFrame(logs)
    df_log.to_csv(filepath, index=False)
    print(f"📝 Logs sauvegardés: {filepath}")


# ===========================
# 5. UTILITAIRES
# ===========================

def generer_rapport_qualite(df, nom_dataset="Dataset", save_path=None):
    """
    Génère un rapport complet de qualité
    """
    rapport = f"""
    {'='*70}
    RAPPORT DE QUALITÉ DES DONNÉES
    {'='*70}
    
    Dataset: {nom_dataset}
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    1. DIMENSIONS
       - Lignes: {len(df):,}
       - Colonnes: {df.shape[1]}
       - Mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    
    2. QUALITÉ GLOBALE
       - Valeurs manquantes: {df.isnull().sum().sum():,} ({(df.isnull().sum().sum() / (len(df) * df.shape[1]) * 100):.2f}%)
       - Doublons: {df.duplicated().sum():,}
    
    3. TYPES DE DONNÉES
    {df.dtypes.value_counts().to_string()}
    
    {'='*70}
    """
    
    print(rapport)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(rapport)
        print(f"\n✅ Rapport sauvegardé: {save_path}")
    
    return rapport


if __name__ == "__main__":
    print("✅ Module cleaning_functions.py chargé avec succès!")
    print("📦 Fonctions disponibles:")
    print("   - analyse_qualite_globale()")
    print("   - analyse_valeurs_manquantes()")
    print("   - detecter_doublons_avances()")
    print("   - analyser_outliers_numeriques()")
    print("   - supprimer_doublons()")
    print("   - imputer_valeurs_manquantes()")
    print("   - normaliser_texte()")
    print("   - plot_valeurs_manquantes()")
    print("   - plot_avant_apres()")
    print("   - generer_rapport_qualite()")
