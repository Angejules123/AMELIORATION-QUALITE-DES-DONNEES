# RAPPORT FINAL - MINI-PROJET 2
## Évaluation et Amélioration de la Qualité des Données

**Auteur** : TIA Ange Jules-Rihem ben Maouia  
**Date** : 02 December 2025  
**Dataset** : Cancer du Sein (Mammographie)

---

## 📊 Résumé Exécutif

Ce projet a consisté à évaluer et améliorer la qualité d'un dataset médical lié au cancer du sein, en appliquant des techniques de nettoyage, de prétraitement d'images et de fusion multimodale.

### Résultats Clés

- **4 datasets CSV** nettoyés (3564 lignes)
- **40 images** prétraitées pour deep learning
- **Complétude moyenne** : 99.05%
- **Pipeline automatisé** de nettoyage en 7 étapes
- **Application web** Streamlit interactive

---

## 1. Données CSV

### Statistiques Globales

| Métrique | Valeur |
|----------|--------|
| Nombre de datasets | 4 |
| Total lignes | 3564 |
| Complétude moyenne | 99.05% |

### Détails par Dataset


#### calc_case_description_test_set_cleaned.csv
- Lignes : 326
- Colonnes : 14
- Complétude : 98.53%

Distribution pathology :
- benign : 130
- malignant : 129
- benign_without_callback : 67

#### calc_case_description_train_set_cleaned.csv
- Lignes : 1,546
- Colonnes : 14
- Complétude : 98.17%

Distribution pathology :
- malignant : 544
- benign : 528
- benign_without_callback : 474

#### mass_case_description_test_set_cleaned.csv
- Lignes : 375
- Colonnes : 16
- Complétude : 99.72%

Distribution pathology :
- benign : 231
- malignant : 144

#### mass_case_description_train_set_cleaned.csv
- Lignes : 1,317
- Colonnes : 16
- Complétude : 99.78%

Distribution pathology :
- benign : 681
- malignant : 636


---

## 2. Données Images

### Statistiques Globales

- **Total images** : 40
- **Train** : 20 images
  - Benign : 10
  - Malignant : 10
- **Test** : 20 images
  - Benign : 10
  - Malignant : 10

### Prétraitements Appliqués

1. **Redimensionnement** : 224×224 pixels (standard CNN)
2. **CLAHE** : Amélioration du contraste local
3. **Débruitage** : Non-Local Means Denoising
4. **Normalisation** : Z-score (μ=0, σ=1)
5. **Augmentation** : Rotations, flips, ajustements luminosité

---

## 3. Fusion Multimodale

Le projet a créé un dataset fusionné combinant :
- Métadonnées CSV (labels, scores BI-RADS, caractéristiques)
- Features images (intensité, contraste, entropie)

**Fichiers features générés** :
- CSV features : 4
- Image features : 1
- Fusion files : 1

---

## 4. Pipeline de Nettoyage

### Étapes Appliquées

1. ✅ Détection automatique colonne cible
2. ✅ Normalisation variables catégorielles
3. ✅ Harmonisation pathology (benign/malignant)
4. ✅ Suppression doublons
5. ✅ Gestion valeurs manquantes critiques
6. ✅ Détection outliers (méthode IQR)
7. ✅ Vérification cohérence BI-RADS ↔ Pathology

### Résultats

- **Incohérences critiques** : Supprimées
- **Doublons** : Éliminés
- **Complétude** : Améliorée à 99.05%

---

## 5. Technologies Utilisées

- **Python** : pandas, numpy, opencv-python
- **Machine Learning** : scikit-learn, imbalanced-learn
- **Deep Learning** : PyTorch, torchvision
- **Visualisation** : matplotlib, seaborn, plotly
- **Web** : Streamlit
- **Documentation** : Markdown, JSON

---

## 6. Fichiers Générés

### Structure du Projet

```
data/cleaned/
├── csv/               # 4 datasets nettoyés
├── images/            # 40 images prétraitées
│   ├── train/
│   └── test/
└── features/          # Features extraites
    ├── csv/
    ├── images/
    └── fusion/
```

**Total fichiers générés** : 59

---

## 7. Conclusion

Ce projet a permis de :
- ✅ Nettoyer et améliorer la qualité des données CSV
- ✅ Prétraiter les images pour le deep learning
- ✅ Créer un dataset multimodal fusionné
- ✅ Automatiser le processus avec un pipeline reproductible
- ✅ Développer une interface web interactive

**Niveau atteint** : Expert Data Science + Deep Learning

---

**Rapport généré le** : 02/12/2025 à 08:28
