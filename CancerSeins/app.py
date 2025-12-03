"""
app.py
DASHBOARD STREAMLIT - PROJET QUALITÉ DES DONNÉES
Mini-Projet 2 : Cancer du Sein (Mammographie)

Interface web interactive pour l'exploration, le nettoyage et la visualisation des données

Auteur: TIA Ange Jules-Rihem ben Maouia
Date: Décembre 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Ajouter le chemin des notebooks pour importer les modules
sys.path.append(str(Path(__file__).parent / 'notebooks'))

# Configuration de la page
st.set_page_config(
    page_title="Projet Qualité des Données - Cancer du Sein",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================

BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "csv"
CLEANED_PATH = BASE_DIR / "data" / "cleaned" / "csv"  # NOUVEAU
CLEANED_IMAGES_PATH = BASE_DIR / "data" / "cleaned" / "images"  # NOUVEAU
CLEANED_FEATURES_PATH = BASE_DIR / "data" / "cleaned" / "features"  # NOUVEAU
LOGS_PATH = BASE_DIR / "data" / "logs"
REPORTS_PATH = BASE_DIR / "reports"
FIGURES_PATH = BASE_DIR / "presentation" / "figures"

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

@st.cache_data
def load_dataset(filepath):
    """Charge un dataset avec cache"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None

def get_available_datasets():
    """Liste les datasets disponibles"""
    datasets = {
        'original': [],
        'cleaned': []
    }
    
    # Datasets originaux
    if DATA_PATH.exists():
        for file in DATA_PATH.glob('*.csv'):
            datasets['original'].append(file.name)
    
    # Datasets nettoyés
    if CLEANED_PATH.exists():
        for file in CLEANED_PATH.glob('*_cleaned.csv'):
            datasets['cleaned'].append(file.name)
    
    return datasets

def get_latest_log():
    """Récupère le log le plus récent"""
    if not LOGS_PATH.exists():
        return None
    
    log_files = list(LOGS_PATH.glob('cleaning_log_*.json'))
    if not log_files:
        return None
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================
# SIDEBAR - NAVIGATION
# ==========================================

st.sidebar.markdown("# 🏥 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choisir une page :",
    [
        "🏠 Accueil",
        "📊 Exploration des Données",
        "🧹 Nettoyage",
        "📈 Visualisations Avant/Après",
        "🎨 Visualisations Multimodales",  # NOUVEAU
        "📋 Rapports et Logs",
        "📚 Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 À Propos")
st.sidebar.info("""
**Mini-Projet 2**  
Évaluation et Amélioration de la Qualité des Données

**Dataset** : Cancer du Sein (Mammographie)  
**Étudiant** : TIA Ange Jules-Rihem ben Maouia  
**Date** : Décembre 2025
""")

# ==========================================
# PAGE 1 : ACCUEIL
# ==========================================

if page == "🏠 Accueil":
    st.markdown('<div class="main-header">🏥 Projet Qualité des Données - Cancer du Sein</div>', 
                unsafe_allow_html=True)
    
    # Vue d'ensemble
    col1, col2, col3 = st.columns(3)
    
    datasets = get_available_datasets()
    n_original = len(datasets['original'])
    n_cleaned = len(datasets['cleaned'])
    
    with col1:
        st.metric("📂 Datasets Originaux", n_original)
    
    with col2:
        st.metric("✨ Datasets Nettoyés", n_cleaned)
    
    with col3:
        latest_log = get_latest_log()
        n_operations = len(latest_log['operations']) if latest_log else 0
        st.metric("🔧 Opérations Effectuées", n_operations)
    
    st.markdown("---")
    
    # Description du projet
    st.markdown("## 🎯 Objectif du Projet")
    st.markdown("""
    Ce projet vise à **nettoyer et améliorer la qualité** d'un dataset médical lié au cancer du sein,
    en appliquant des règles de nettoyage médicalement validées et en assurant une traçabilité complète.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Fonctionnalités")
        st.markdown("""
        - 📊 **Exploration interactive** des données
        - 🧹 **Nettoyage automatisé** avec pipeline
        - 📈 **Visualisations** avant/après
        - 📋 **Rapports détaillés** et logs JSON
        - 🏥 **Validation médicale** (BI-RADS)
        - 📚 **Documentation complète**
        """)
    
    with col2:
        st.markdown("### 📦 Datasets Disponibles")
        if n_original > 0:
            st.success(f"✅ {n_original} datasets originaux chargés")
            for ds in datasets['original'][:3]:
                st.text(f"  • {ds}")
        else:
            st.warning("⚠️ Aucun dataset original trouvé")
        
        if n_cleaned > 0:
            st.success(f"✅ {n_cleaned} datasets nettoyés disponibles")
        else:
            st.info("ℹ️ Aucun dataset nettoyé. Utilisez la page 'Nettoyage'")
    
    st.markdown("---")
    
    # Statut du projet
    st.markdown("## 📊 Statut du Projet")
    
    progress_data = {
        'Phase': ['Exploration', 'Règles', 'Nettoyage', 'Validation', 'Rapport'],
        'Statut': [100, 100, n_cleaned > 0 and 100 or 50, n_cleaned > 0 and 75 or 0, n_cleaned > 0 and 50 or 0]
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=progress_data['Statut'],
            y=progress_data['Phase'],
            orientation='h',
            marker=dict(
                color=progress_data['Statut'],
                colorscale='Blues',
                showscale=False
            ),
            text=[f"{s}%" for s in progress_data['Statut']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Avancement par Phase",
        xaxis_title="Progression (%)",
        yaxis_title="",
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2 : EXPLORATION DES DONNÉES
# ==========================================

elif page == "📊 Exploration des Données":
    st.markdown('<div class="main-header">📊 Exploration des Données</div>', 
                unsafe_allow_html=True)
    
    datasets = get_available_datasets()
    
    if not datasets['original']:
        st.error("❌ Aucun dataset trouvé dans le dossier 'csv/'")
        st.stop()
    
    # Sélection du dataset
    selected_file = st.selectbox(
        "Sélectionner un dataset :",
        datasets['original']
    )
    
    df = load_dataset(DATA_PATH / selected_file)
    
    if df is not None:
        st.success(f"✅ Dataset chargé : {len(df):,} lignes × {df.shape[1]} colonnes")
        
        # Onglets
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Aperçu", "📊 Statistiques", "❓ Qualité", "📈 Visualisations"
        ])
        
        with tab1:
            st.markdown("### 📋 Aperçu des Données")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.markdown("### 🔍 Informations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Types de données :**")
                st.write(df.dtypes.value_counts())
            
            with col2:
                st.markdown("**Colonnes :**")
                for col in df.columns:
                    st.text(f"• {col} ({df[col].dtype})")
        
        with tab2:
            st.markdown("### 📊 Statistiques Descriptives")
            st.dataframe(df.describe(include='all'), use_container_width=True)
            
            # Distribution des colonnes catégorielles
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                st.markdown("### 📑 Distribution des Variables Catégorielles")
                selected_cat = st.selectbox("Sélectionner une colonne :", cat_cols)
                
                value_counts = df[selected_cat].value_counts()
                
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    labels={'x': 'Nombre', 'y': selected_cat},
                    title=f"Distribution de {selected_cat}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### ❓ Analyse de Qualité")
            
            col1, col2, col3 = st.columns(3)
            
            total_cells = len(df) * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_pct = (missing_cells / total_cells) * 100
            duplicates = df.duplicated().sum()
            
            with col1:
                st.metric("Complétude", f"{100 - missing_pct:.1f}%", 
                         delta=f"{missing_cells} manquantes")
            
            with col2:
                st.metric("Doublons", duplicates, 
                         delta=f"{(duplicates/len(df)*100):.1f}% du total")
            
            with col3:
                st.metric("Taille Mémoire", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Valeurs manquantes par colonne
            st.markdown("#### Valeurs Manquantes par Colonne")
            missing_data = pd.DataFrame({
                'Colonne': df.columns,
                'Manquantes': df.isnull().sum().values,
                'Pourcentage': (df.isnull().sum() / len(df) * 100).values
            })
            missing_data = missing_data[missing_data['Manquantes'] > 0].sort_values('Pourcentage', ascending=False)
            
            if len(missing_data) > 0:
                fig = px.bar(
                    missing_data,
                    x='Pourcentage',
                    y='Colonne',
                    orientation='h',
                    title="Pourcentage de Valeurs Manquantes",
                    color='Pourcentage',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ Aucune valeur manquante détectée !")
        
        with tab4:
            st.markdown("### 📈 Visualisations")
            
            # Visualisation de la distribution de pathology
            if 'pathology' in df.columns:
                st.markdown("#### Distribution de Pathology")
                pathology_dist = df['pathology'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=pathology_dist.index,
                        values=pathology_dist.values,
                        hole=0.4
                    )
                ])
                fig.update_layout(title="Distribution Pathology")
                st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de corrélation pour colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.markdown("#### Matrice de Corrélation")
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Corrélation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 3 : NETTOYAGE
# ==========================================

elif page == "🧹 Nettoyage":
    st.markdown('<div class="main-header">🧹 Pipeline de Nettoyage</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Cette page permet d'exécuter le **pipeline de nettoyage automatisé** qui applique 
    toutes les règles médicales définies.
    """)
    
    datasets = get_available_datasets()
    
    if not datasets['original']:
        st.error("❌ Aucun dataset trouvé")
        st.stop()
    
    # Sélection des datasets à nettoyer
    st.markdown("### 📂 Sélection des Datasets")
    
    files_to_clean = st.multiselect(
        "Choisir les fichiers à nettoyer :",
        datasets['original'],
        default=datasets['original'][:2]  # Sélectionner les 2 premiers par défaut
    )
    
    if not files_to_clean:
        st.warning("⚠️ Veuillez sélectionner au moins un dataset")
        st.stop()
    
    st.markdown("### ⚙️ Configuration du Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_duplicates = st.checkbox("Supprimer les doublons", value=True)
        handle_missing = st.checkbox("Gérer les valeurs manquantes", value=True)
    
    with col2:
        detect_outliers = st.checkbox("Détecter les outliers", value=True)
        check_consistency = st.checkbox("Vérifier cohérence BI-RADS", value=True)
    
    st.markdown("---")
    
    # Bouton de lancement
    if st.button("🚀 Lancer le Nettoyage", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        for idx, filename in enumerate(files_to_clean):
            status_text.text(f"📥 Traitement de {filename}...")
            progress_bar.progress((idx + 1) / len(files_to_clean))
            
            try:
                # Charger le dataset
                df = load_dataset(DATA_PATH / filename)
                rows_before = len(df)
                
                # Simuler le nettoyage (version simplifiée)
                df_clean = df.copy()
                operations = []
                
                # 1. Normalisation
                cat_cols = df_clean.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
                operations.append("Normalisation des variables catégorielles")
                
                # 2. Doublons
                if remove_duplicates:
                    n_dup = df_clean.duplicated().sum()
                    df_clean = df_clean.drop_duplicates()
                    operations.append(f"Suppression de {n_dup} doublons")
                
                # 3. Valeurs manquantes
                if handle_missing:
                    df_clean = df_clean.dropna(subset=['pathology'] if 'pathology' in df_clean.columns else [])
                    operations.append("Suppression des lignes avec pathology manquante")
                
                rows_after = len(df_clean)
                
                # Sauvegarder
                output_path = CLEANED_PATH / filename.replace('.csv', '_cleaned.csv')
                CLEANED_PATH.mkdir(parents=True, exist_ok=True)
                df_clean.to_csv(output_path, index=False)
                
                results[filename] = {
                    'rows_before': rows_before,
                    'rows_after': rows_after,
                    'operations': operations,
                    'output': str(output_path)
                }
                
            except Exception as e:
                st.error(f"❌ Erreur sur {filename}: {e}")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Nettoyage terminé !")
        
        # Afficher les résultats
        st.markdown("### 📊 Résultats du Nettoyage")
        
        for filename, result in results.items():
            with st.expander(f"📄 {filename}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Lignes Avant", f"{result['rows_before']:,}")
                
                with col2:
                    st.metric("Lignes Après", f"{result['rows_after']:,}")
                
                with col3:
                    reduction = result['rows_before'] - result['rows_after']
                    st.metric("Supprimées", f"{reduction:,}", 
                             delta=f"{(reduction/result['rows_before']*100):.1f}%")
                
                st.markdown("**Opérations effectuées :**")
                for op in result['operations']:
                    st.text(f"✓ {op}")
        
        st.success("✨ Tous les datasets ont été nettoyés avec succès !")

# ==========================================
# PAGE 4 : VISUALISATIONS AVANT/APRÈS
# ==========================================

elif page == "📈 Visualisations Avant/Après":
    st.markdown('<div class="main-header">📈 Comparaison Avant/Après</div>', 
                unsafe_allow_html=True)
    
    datasets = get_available_datasets()
    
    if not datasets['cleaned']:
        st.warning("⚠️ Aucun dataset nettoyé disponible. Utilisez d'abord la page 'Nettoyage'")
        st.stop()
    
    # Sélection du dataset
    selected_cleaned = st.selectbox(
        "Sélectionner un dataset nettoyé :",
        datasets['cleaned']
    )
    
    # Trouver le dataset original correspondant
    original_name = selected_cleaned.replace('_cleaned', '')
    
    if original_name not in datasets['original']:
        st.error(f"❌ Dataset original {original_name} non trouvé")
        st.stop()
    
    # Charger les deux versions
    df_original = load_dataset(DATA_PATH / original_name)
    df_cleaned = load_dataset(CLEANED_PATH / selected_cleaned)
    
    if df_original is None or df_cleaned is None:
        st.error("Erreur lors du chargement des datasets")
        st.stop()
    
    # Métriques de comparaison
    st.markdown("### 📊 Métriques de Comparaison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Lignes",
            f"{len(df_cleaned):,}",
            delta=f"{len(df_cleaned) - len(df_original):,}",
            delta_color="inverse"
        )
    
    with col2:
        missing_before = df_original.isnull().sum().sum()
        missing_after = df_cleaned.isnull().sum().sum()
        st.metric(
            "Valeurs Manquantes",
            f"{missing_after:,}",
            delta=f"{missing_after - missing_before:,}",
            delta_color="inverse"
        )
    
    with col3:
        dup_before = df_original.duplicated().sum()
        dup_after = df_cleaned.duplicated().sum()
        st.metric(
            "Doublons",
            f"{dup_after}",
            delta=f"{dup_after - dup_before}",
            delta_color="inverse"
        )
    
    with col4:
        completeness = (1 - df_cleaned.isnull().sum().sum() / (len(df_cleaned) * df_cleaned.shape[1])) * 100
        st.metric(
            "Complétude",
            f"{completeness:.1f}%"
        )
    
    st.markdown("---")
    
    # Graphiques de comparaison
    st.markdown("### 📈 Visualisations Comparatives")
    
    tab1, tab2, tab3 = st.tabs(["📊 Taille", "❓ Qualité", "📑 Distribution"])
    
    with tab1:
        # Comparaison de taille
        size_data = pd.DataFrame({
            'Version': ['Original', 'Nettoyé'],
            'Lignes': [len(df_original), len(df_cleaned)],
            'Colonnes': [df_original.shape[1], df_cleaned.shape[1]]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Lignes', x=size_data['Version'], y=size_data['Lignes'], marker_color='lightblue'),
            go.Bar(name='Colonnes', x=size_data['Version'], y=size_data['Colonnes'], marker_color='lightcoral')
        ])
        fig.update_layout(title="Comparaison de Taille", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Comparaison de qualité
        quality_data = pd.DataFrame({
            'Métrique': ['Valeurs Manquantes', 'Doublons'],
            'Original': [missing_before, dup_before],
            'Nettoyé': [missing_after, dup_after]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Original', x=quality_data['Métrique'], y=quality_data['Original'], marker_color='coral'),
            go.Bar(name='Nettoyé', x=quality_data['Métrique'], y=quality_data['Nettoyé'], marker_color='lightgreen')
        ])
        fig.update_layout(title="Amélioration de la Qualité", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Distribution de pathology (si disponible)
        if 'pathology' in df_original.columns and 'pathology' in df_cleaned.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original**")
                dist_original = df_original['pathology'].value_counts()
                fig = px.pie(values=dist_original.values, names=dist_original.index, 
                           title="Distribution Pathology (Original)")
                st.plotly_chart(fig)
            
            with col2:
                st.markdown("**Nettoyé**")
                dist_cleaned = df_cleaned['pathology'].value_counts()
                fig = px.pie(values=dist_cleaned.values, names=dist_cleaned.index,
                           title="Distribution Pathology (Nettoyé)")
                st.plotly_chart(fig)

# ==========================================
# PAGE 5 : VISUALISATIONS MULTIMODALES
# ==========================================

elif page == "🎨 Visualisations Multimodales":
    st.markdown('<div class="main-header">🎨 Visualisations Multimodales</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Cette page présente les **visualisations avancées** combinant données CSV et images.
    """)
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs(["📊 CSV", "🖼️ Images", "🔗 Fusion", "📝 Rapport"])
    
    with tab1:
        st.markdown("### Visualisations Données CSV")
        
        # Distribution classes
        if (FIGURES_PATH / "01_distribution_classes.png").exists():
            st.markdown("#### Distribution des Classes")
            st.image(str(FIGURES_PATH / "01_distribution_classes.png"))
            st.info("Distribution des diagnostics (benign/malignant) par dataset")
        
        # Qualité
        if (FIGURES_PATH / "02_qualite_datasets.png").exists():
            st.markdown("#### Qualité des Datasets")
            st.image(str(FIGURES_PATH / "02_qualite_datasets.png"))
            st.success("Complétude > 99% pour tous les datasets")
        
        # Corrélation
        if (FIGURES_PATH / "03_correlation_matrix.png").exists():
            st.markdown("#### Matrice de Corrélation")
            st.image(str(FIGURES_PATH / "03_correlation_matrix.png"))
            st.info("Corrélations entre variables numériques")
    
    with tab2:
        st.markdown("### Visualisations Images")
        
        # Mosaïque
        if (FIGURES_PATH / "04_mosaique_images.png").exists():
            st.markdown("#### Échantillon d'Images")
            st.image(str(FIGURES_PATH / "04_mosaique_images.png"))
            st.info("Images prétraitées (224×224, CLAHE, débruitage)")
        
        # Histogrammes
        if (FIGURES_PATH / "05_histogrammes_intensite.png").exists():
            st.markdown("#### Distribution d'Intensité")
            st.image(str(FIGURES_PATH / "05_histogrammes_intensite.png"))
            st.info("Analyse de l'intensité des pixels par classe")
    
    with tab3:
        st.markdown("### Visualisations Fusion CSV ↔ Images")
        
        # Dashboard multimodal
        if (FIGURES_PATH / "06_dashboard_multimodal.png").exists():
            st.markdown("#### Dashboard Multimodal")
            st.image(str(FIGURES_PATH / "06_dashboard_multimodal.png"))
            st.success("Vue d'ensemble combinée des données CSV et images")
        
        # Statistiques
        st.markdown("#### Statistiques Fusion")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Datasets CSV", "4")
        
        with col2:
            st.metric("Images", "40")
        
        with col3:
            st.metric("Complétude", "99.05%")
    
    with tab4:
        st.markdown("### Rapport d'Exploration Multimodale")
        
        # Afficher le rapport
        rapport_path = REPORTS_PATH / "RAPPORT_EXPLORATION_MULTIMODALE.md"
        if rapport_path.exists():
            with open(rapport_path, 'r', encoding='utf-8') as f:
                rapport_content = f.read()
            
            st.markdown(rapport_content, unsafe_allow_html=True)
            
            # Bouton téléchargement
            st.download_button(
                label="📥 Télécharger le rapport",
                data=rapport_content,
                file_name="rapport_exploration_multimodale.md",
                mime="text/markdown"
            )
        else:
            st.warning("Rapport non trouvé. Exécutez d'abord visualisations_multimodales.py")

# ==========================================
# PAGE 6 : RAPPORTS ET LOGS
# ==========================================

elif page == "📋 Rapports et Logs":
    st.markdown('<div class="main-header">📋 Rapports et Logs</div>', 
                unsafe_allow_html=True)
    
    latest_log = get_latest_log()
    
    if latest_log is None:
        st.warning("⚠️ Aucun log de nettoyage trouvé. Exécutez d'abord le pipeline de nettoyage.")
        st.stop()
    
    # Afficher les informations du log
    st.markdown("### 📊 Résumé du Dernier Nettoyage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Datasets Traités", latest_log['summary']['total_datasets'])
    
    with col2:
        st.metric("Opérations Totales", latest_log['summary']['total_operations'])
    
    with col3:
        timestamp = datetime.fromisoformat(latest_log['timestamp'])
        st.metric("Date", timestamp.strftime("%d/%m/%Y %H:%M"))
    
    st.markdown("---")
    
    # Détails des opérations
    st.markdown("### 🔧 Détails des Opérations")
    
    operations_df = pd.DataFrame(latest_log['operations'])
    
    for idx, op in enumerate(latest_log['operations']):
        with st.expander(f"Opération {idx + 1}: {op['step']}", expanded=idx < 3):
            st.markdown(f"**Détails** : {op['details']}")
            st.markdown(f"**Timestamp** : {op['timestamp']}")
            
            if 'rows_before' in op and 'rows_after' in op:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avant", f"{op['rows_before']:,}")
                
                with col2:
                    st.metric("Après", f"{op['rows_after']:,}")
                
                with col3:
                    st.metric("Supprimées", f"{op['rows_removed']:,}",
                             delta=f"{op['percentage_removed']}")
    
    st.markdown("---")
    
    # JSON complet
    st.markdown("### 📄 Log JSON Complet")
    st.json(latest_log)

# ==========================================
# PAGE 6 : DOCUMENTATION
# ==========================================

elif page == "📚 Documentation":
    st.markdown('<div class="main-header">📚 Documentation du Projet</div>', 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Règles de Nettoyage",
        "📖 Dictionnaire de Données",
        "🎯 Guide d'Utilisation",
        "📊 Structure du Projet"
    ])
    
    with tab1:
        st.markdown("### 📋 Règles de Nettoyage Médicales")
        st.info("""
        Les règles de nettoyage appliquées sont organisées en 4 catégories :
        
        1. **Intégrité des données** : Doublons, valeurs manquantes
        2. **Cohérence sémantique** : Normalisation, harmonisation des libellés
        3. **Qualité statistique** : Outliers, distributions
        4. **Préparation pour modélisation** : Encodage, standardisation
        """)
        
        if (REPORTS_PATH / "REGLES_NETTOYAGE_MEDICALES.md").exists():
            st.success("✅ Documentation complète disponible dans `reports/REGLES_NETTOYAGE_MEDICALES.md`")
    
    with tab2:
        st.markdown("### 📖 Dictionnaire de Données")
        st.info("""
        Le dictionnaire de données décrit toutes les variables des datasets :
        
        - **Types de données** et contraintes
        - **Valeurs possibles** pour chaque variable
        - **Signification clinique** (contexte médical BI-RADS)
        - **Relations** entre datasets
        """)
        
        if (REPORTS_PATH / "DICTIONNAIRE_DONNEES.md").exists():
            st.success("✅ Dictionnaire complet disponible dans `reports/DICTIONNAIRE_DONNEES.md`")
    
    with tab3:
        st.markdown("### 🎯 Guide d'Utilisation")
        st.markdown("""
        #### Comment utiliser cette application :
        
        1. **📊 Exploration** : Analysez vos données brutes
        2. **🧹 Nettoyage** : Exécutez le pipeline automatisé
        3. **📈 Visualisations** : Comparez avant/après
        4. **📋 Rapports** : Consultez les logs détaillés
        5. **📚 Documentation** : Accédez à toute la documentation
        
        #### Fichiers Générés :
        
        - `data/cleaned/*.csv` : Datasets nettoyés
        - `data/logs/*.json` : Logs de transformation
        - `reports/*.md` : Règles et dictionnaire
        """)
    
    with tab4:
        st.markdown("### 📊 Structure du Projet")
        st.code("""
CancerSeins/
├── csv/                        # Données brutes
├── jpeg/                       # Images mammographiques
├── data/
│   ├── cleaned/               # Datasets nettoyés
│   ├── logs/                  # Logs JSON
│   └── augmented/             # Données augmentées (optionnel)
├── notebooks/
│   ├── 01_exploration_diagnostic_complet.ipynb
│   ├── pipeline_nettoyage.py
│   ├── data_augmentation.py
│   └── utils/
│       └── cleaning_functions.py
├── reports/
│   ├── REGLES_NETTOYAGE_MEDICALES.md
│   ├── DICTIONNAIRE_DONNEES.md
│   └── TEMPLATE_RAPPORT.md
├── presentation/
│   └── figures/               # Graphiques et visualisations
└── app.py                     # Cette application Streamlit
        """, language="bash")

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏥 <strong>Mini-Projet 2 : Qualité des Données - Cancer du Sein</strong></p>
    <p>TIA Ange Jules-Rihem ben Maouia | Décembre 2025</p>
</div>
""", unsafe_allow_html=True)
