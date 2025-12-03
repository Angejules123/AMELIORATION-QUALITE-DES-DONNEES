# DICTIONNAIRE DE DONNÉES - CANCER DU SEIN (MAMMOGRAPHIE)

## Datasets : Calcifications et Masses Tumorales

**Projet** : Mini-Projet 2 - Évaluation et Amélioration de la Qualité des Données  
**Étudiant** : TIA Ange Jules-Rihem ben Maouia  
**Date** : Décembre 2025

---

## 📋 VUE D'ENSEMBLE

Ce dictionnaire décrit toutes les variables des datasets :

- `calc_case_description_train_set.csv` - Cas de calcifications
- `calc_case_description_test_set.csv`
- `mass_case_description_train_set.csv` - Cas de masses
- `mass_case_description_test_set.csv`

---

## 📊 TABLE PRINCIPALE : CALCIFICATIONS

| Nom de Colonne           | Type          | Description                      | Valeurs Possibles                                                                                                                                 | Contraintes      | Critique |
| ------------------------ | ------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | -------- |
| **patient_id**           | `integer`     | Identifiant unique du patient    | Entier positif                                                                                                                                    | Non NULL, Unique | ⚠️       |
| **breast_density**       | `categorical` | Densité mammaire selon BI-RADS   | A, B, C, D                                                                                                                                        | Non NULL         | ✅       |
| **left_or_right_breast** | `categorical` | Sein concerné                    | LEFT, RIGHT                                                                                                                                       | Non NULL         | ✅       |
| **image_view**           | `categorical` | Vue radiologique                 | CC, MLO                                                                                                                                           | Non NULL         | ⚠️       |
| **abnormality_id**       | `integer`     | Identifiant de l'anomalie        | Entier positif                                                                                                                                    | Non NULL         | ⚠️       |
| **abnormality_type**     | `categorical` | Type d'anomalie                  | calcification                                                                                                                                     | Fixe             | ✅       |
| **calc_type**            | `categorical` | Type de calcification            | pleomorphic, amorphous, coarse, round_and_regular, lucent_centered, eggshell, skin, vascular, suture, dystrophic, punctate, fine_linear_branching | Non NULL         | ✅       |
| **calc_distribution**    | `categorical` | Distribution des calcifications  | clustered, linear, segmental, regional, diffusely_scattered                                                                                       | Non NULL         | ✅       |
| **assessment**           | `integer`     | Score BI-RADS                    | 0, 1, 2, 3, 4, 5                                                                                                                                  | Non NULL, 0-5    | ✅✅     |
| **pathology**            | `categorical` | Diagnostic final confirmé        | BENIGN, MALIGNANT, BENIGN_WITHOUT_CALLBACK                                                                                                        | Non NULL         | ✅✅     |
| **subtlety**             | `integer`     | Degré de difficulté de détection | 1, 2, 3, 4, 5                                                                                                                                     | 1-5              | ⚠️       |

### Légende

- ✅✅ : **Critique** - Aucune valeur manquante tolérée
- ✅ : **Importante** - Peu de valeurs manquantes acceptées
- ⚠️ : **Métadonnée** - Valeurs manquantes tolérées selon contexte

---

## 📊 TABLE PRINCIPALE : MASSES

| Nom de Colonne           | Type          | Description                      | Valeurs Possibles                                                | Contraintes      | Critique |
| ------------------------ | ------------- | -------------------------------- | ---------------------------------------------------------------- | ---------------- | -------- |
| **patient_id**           | `integer`     | Identifiant unique du patient    | Entier positif                                                   | Non NULL, Unique | ⚠️       |
| **breast_density**       | `categorical` | Densité mammaire selon BI-RADS   | A, B, C, D                                                       | Non NULL         | ✅       |
| **left_or_right_breast** | `categorical` | Sein concerné                    | LEFT, RIGHT                                                      | Non NULL         | ✅       |
| **image_view**           | `categorical` | Vue radiologique                 | CC, MLO                                                          | Non NULL         | ⚠️       |
| **abnormality_id**       | `integer`     | Identifiant de l'anomalie        | Entier positif                                                   | Non NULL         | ⚠️       |
| **abnormality_type**     | `categorical` | Type d'anomalie                  | mass                                                             | Fixe             | ✅       |
| **mass_shape**           | `categorical` | Forme de la masse                | round, oval, lobulated, irregular, architectural_distortion      | Non NULL         | ✅       |
| **mass_margins**         | `categorical` | Caractéristiques des bords       | circumscribed, microlobulated, obscured, ill-defined, spiculated | Non NULL         | ✅✅     |
| **assessment**           | `integer`     | Score BI-RADS                    | 0, 1, 2, 3, 4, 5                                                 | Non NULL, 0-5    | ✅✅     |
| **pathology**            | `categorical` | Diagnostic final confirmé        | BENIGN, MALIGNANT, BENIGN_WITHOUT_CALLBACK                       | Non NULL         | ✅✅     |
| **subtlety**             | `integer`     | Degré de difficulté de détection | 1, 2, 3, 4, 5                                                    | 1-5              | ⚠️       |

---

## 📖 DESCRIPTIONS DÉTAILLÉES DES VARIABLES

### 1. Identifiants

#### `patient_id`

- **Nature** : Identifiant anonymisé du patient
- **Format** : Entier positif unique
- **Utilité** : Traçabilité et jointure avec autres datasets
- **Exemple** : `12345`, `67890`

#### `abnormality_id`

- **Nature** : Identifiant de la lésion/anomalie détectée
- **Format** : Entier positif
- **Utilité** : Plusieurs anomalies peuvent exister pour un même patient
- **Exemple** : `1`, `2`, `3`

---

### 2. Localisation Anatomique

#### `left_or_right_breast`

- **Nature** : Côté du sein concerné
- **Valeurs** :
  - `LEFT` : Sein gauche
  - `RIGHT` : Sein droit
- **Utilité clinique** : Les lésions bilatérales sont plus rares et suspectes

#### `image_view`

- **Nature** : Vue radiologique utilisée
- **Valeurs** :
  - `CC` : Crânio-caudale (vue de dessus)
  - `MLO` : Médio-latérale oblique (vue de côté)
- **Utilité** : Détection complémentaire, certaines lésions sont mieux visibles sur certaines vues

---

### 3. Caractérisation Tissulaire

#### `breast_density`

- **Nature** : Densité du tissu mammaire selon classification BI-RADS
- **Valeurs** :
  - `A` : Presque entièrement graisseuse (< 25% de tissu dense)
  - `B` : Densités fibroglandulaires dispersées (25-50%)
  - `C` : Tissu dense hétérogène (51-75%) - peut masquer des lésions
  - `D` : Tissu extrêmement dense (> 75%) - réduit la sensibilité de la mammographie
- **Impact clinique** : Densités C et D rendent la détection plus difficile

---

### 4. Caractéristiques des Calcifications

#### `calc_type`

- **Nature** : Type morphologique de calcification
- **Valeurs et Signification Clinique** :

| Valeur                  | Signification                            | Association Cancer   | Typologie BI-RADS  |
| ----------------------- | ---------------------------------------- | -------------------- | ------------------ |
| `fine_linear_branching` | Calcifications fines linéaires ramifiées | **ÉLEVÉE** (>80%)    | Hautement suspect  |
| `pleomorphic`           | Forme et taille hétérogènes              | **MODÉRÉE** (40-60%) | Suspect            |
| `amorphous`             | Forme indistincte                        | **MODÉRÉE** (20-40%) | Intermédiaire      |
| `coarse`                | Grossières (>0.5mm)                      | **FAIBLE** (<5%)     | Typiquement bénin  |
| `punctate`              | Punctiformes (<0.5mm)                    | **FAIBLE** (<10%)    | Probablement bénin |
| `round_and_regular`     | Rondes et régulières                     | **FAIBLE** (<5%)     | Typiquement bénin  |
| `lucent_centered`       | Centre clair                             | **TRÈS FAIBLE**      | Bénin              |
| `eggshell`              | En coquille d'œuf                        | **TRÈS FAIBLE**      | Bénin              |
| `skin`                  | Cutanées                                 | **AUCUNE**           | Bénin              |
| `vascular`              | Vasculaires                              | **AUCUNE**           | Bénin              |
| `suture`                | Post-chirurgicales                       | **AUCUNE**           | Bénin              |
| `dystrophic`            | Dystrophiques (nécrose graisseuse)       | **AUCUNE**           | Bénin              |

#### `calc_distribution`

- **Nature** : Répartition spatiale des calcifications
- **Valeurs et Signification Clinique** :

| Valeur                | Signification                 | Association Cancer    | Description                           |
| --------------------- | ----------------------------- | --------------------- | ------------------------------------- |
| `diffusely_scattered` | Dispersées de manière diffuse | **TRÈS FAIBLE** (<2%) | Dans tout le sein, bilatéral possible |
| `regional`            | Répartition régionale         | **FAIBLE** (5-10%)    | Grande zone du sein                   |
| `clustered`           | Groupées en amas              | **MODÉRÉE** (20-40%)  | Au moins 5 calcifications dans 1cm²   |
| `linear`              | Distribution linéaire         | **ÉLEVÉE** (50-70%)   | Suivent un canal galactophore         |
| `segmental`           | Distribution segmentaire      | **ÉLEVÉE** (60-80%)   | Suivent un territoire canalaire       |

---

### 5. Caractéristiques des Masses

#### `mass_shape`

- **Nature** : Forme morphologique de la masse
- **Valeurs et Signification Clinique** :

| Valeur                     | Signification             | Association Cancer   | Raison                                   |
| -------------------------- | ------------------------- | -------------------- | ---------------------------------------- |
| `round`                    | Forme ronde               | **FAIBLE** (<10%)    | Généralement bénin (kyste, fibroadénome) |
| `oval`                     | Forme ov ale              | **FAIBLE** (<15%)    | Généralement bénin                       |
| `lobulated`                | Lobulée (avec lobes)      | **MODÉRÉE** (30-50%) | Ambiguë, nécessite investigation         |
| `irregular`                | Forme irrégulière         | **ÉLEVÉE** (>60%)    | Fortement suspecte                       |
| `architectural_distortion` | Distorsion architecturale | **ÉLEVÉE** (>70%)    | Très suspecte, tissu désorganisé         |

#### `mass_margins`

- **Nature** : Caractéristiques des contours/bords de la masse
- **Valeurs et Signification Clinique** :

| Valeur           | Signification                        | Association Cancer     | Raison                        |
| ---------------- | ------------------------------------ | ---------------------- | ----------------------------- |
| `circumscribed`  | Bien délimitée, nette                | **TRÈS FAIBLE** (<5%)  | Typique des lésions bénignes  |
| `obscured`       | Partiellement cachée par tissu dense | **FAIBLE** (10-20%)    | Difficulté technique          |
| `microlobulated` | Petites ondulations                  | **MODÉRÉE** (40-60%)   | Suspect                       |
| `ill-defined`    | Mal définie, floue                   | **ÉLEVÉE** (60-75%)    | Infiltration suspecte         |
| `spiculated`     | Bords spiculés (en rayons)           | **TRÈS ÉLEVÉE** (>80%) | Fortement évocateur de cancer |

---

### 6. Évaluation Radiologique

#### `assessment`

- **Nature** : Score BI-RADS (Breast Imaging-Reporting and Data System)
- **Valeurs** : 0, 1, 2, 3, 4, 5
- **Signification Clinique** :

| Score | Signification                  | Risque Malignité | Action Recommandée                  |
| ----- | ------------------------------ | ---------------- | ----------------------------------- |
| **0** | Évaluation incomplète          | Non applicable   | Examens complémentaires nécessaires |
| **1** | Négatif                        | 0%               | Dépistage de routine (1-2 ans)      |
| **2** | Bénin                          | 0%               | Dépistage de routine (1-2 ans)      |
| **3** | Probablement bénin             | < 2%             | Surveillance rapprochée (6 mois)    |
| **4** | Anomalie suspecte              | 2-95%            | Biopsie recommandée                 |
| **5** | Hautement suspect de malignité | ≥ 95%            | Biopsie urgente                     |

**Note** : BI-RADS 4 est parfois subdivisé en 4A (2-10%), 4B (10-50%), 4C (50-95%)

---

### 7. Diagnostic Final

#### `pathology`

- **Nature** : Résultat histopathologique (gold standard)
- **Valeurs** :
  - `BENIGN` : Lésion bénigne confirmée histologiquement
  - `MALIGNANT` : Cancer confirmé histologiquement
  - `BENIGN_WITHOUT_CALLBACK` : Bénin sans nécessité de suivi rapproché

**Justification médicale** :

- Seule la **biopsie** avec analyse histologique peut confirmer définitivement
- C'est la variable cible pour les modèles prédictifs
- La mammographie (BI-RADS) est un outil diagnostique, pas un diagnostic final

---

### 8. Métadonnées Qualitatives

#### `subtlety`

- **Nature** : Degré de difficulté de détection de la lésion
- **Échelle** : 1 à 5
- **Signification** :
  - `1` : Très subtile, très difficile à détecter
  - `2` : Subtile
  - `3` : Moyennement visible
  - `4` : Relativement évidente
  - `5` : Très évidente, facilement détectable

**Utilité** :

- Évaluer la performance des radiologues
- Identifier les cas complexes
- Pondération pour l'entraînement de modèles

---

## 🔗 RELATIONS ENTRE DATASETS

### Jointures Possibles

```sql
-- Relation Patient → Cas
patient_id (clé primaire)

-- Relation Patient → Images
patient_id + abnormality_id (clé composite)

-- Relation Cas → Métadonnées DICOM
SeriesInstanceUID (depuis meta.csv)
```

### Schéma Relationnel Simplifié

```
[Patient] 1----N [Abnormality] 1----N [Images DICOM]
    |                 |
patient_id      abnormality_id
```

---

## 📏 RÈGLES DE VALIDATION

### Contraintes d'Intégrité

1. **Unicité** : `patient_id` doit être unique par ligne (sauf si plusieurs anomalies)
2. **Cohérence BI-RADS ↔ Pathology** : Voir règles médicales
3. **Cohérence Morphologie ↔ Pathology** : Voir matrice de cohérence
4. **Plages de valeurs** :
   - `assessment` : [0-5]
   - `subtlety` : [1-5]
   - `breast_density` : {A, B, C, D}

---

## 📊 STATISTIQUES DESCRIPTIVES (Exemple)

### Distribution Typique Attendue

| Variable           | Catégorie | Fréquence Attendue |
| ------------------ | --------- | ------------------ |
| **pathology**      | BENIGN    | 70-80%             |
|                    | MALIGNANT | 20-30%             |
| **breast_density** | A         | 10%                |
|                    | B         | 40%                |
|                    | C         | 40%                |
|                    | D         | 10%                |
| **assessment**     | 1-2       | 40-50%             |
|                    | 3         | 20-30%             |
|                    | 4-5       | 20-30%             |

**Note** : Ces valeurs sont indicatives, les vraies distributions varient selon la cohorte

---

## 🔄 TRANSFORMATIONS APPLIQUÉES

### Variables Dérivées Possibles

| Variable Dérivée       | Formule/Source                           | Utilité             |
| ---------------------- | ---------------------------------------- | ------------------- |
| `age_group`            | Tranches d'âge (si age disponible)       | Analyse par cohorte |
| `high_risk_morphology` | margin="spiculated" OR shape="irregular" | Feature engineering |
| `cancer_probability`   | Basé sur BI-RADS                         | Score de risque     |
| `consistency_flag`     | Cohérence BI-RADS ↔ Pathology            | Contrôle qualité    |

---

## 📖 RÉFÉRENCES

1. **American College of Radiology** - ACR BI-RADS® Atlas, 5th Edition
2. **D'Orsi CJ, Sickles EA, Mendelson EB, Morris EA** - Breast Imaging Reporting and Data System (2013)
3. **CBIS-DDSM Dataset** - Curated Breast Imaging Subset of DDSM (Lee et al., 2017)

---

**Document créé le** : 01 Décembre 2025  
**Dernière mise à jour** : 01 Décembre 2025  
**Version** : 1.0  
**Statut** : ✅ Validé pour utilisation
