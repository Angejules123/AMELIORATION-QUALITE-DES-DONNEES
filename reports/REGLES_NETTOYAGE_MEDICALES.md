# RÈGLES DE NETTOYAGE DES DONNÉES - CONTEXTE MÉDICAL

## Dataset : Cancer du Sein (Mammographie)

**Étudiant** : TIA Ange Jules-Rihem ben Maouia  
**Date** : Décembre 2025  
**Datasets concernés** : `calc_case_description_train_set.csv`, `mass_case_description_train_set.csv`

---

## 📋 VUE D'ENSEMBLE

Les règles de nettoyage sont organisées en **4 catégories** :

1. ✅ **Intégrité des données**
2. ✅ **Cohérence sémantique et médicale**
3. ✅ **Qualité statistique**
4. ✅ **Préparation pour modélisation**

---

## 1. INTÉGRITÉ DES DONNÉES

### 1.1. Détection et Suppression des Doublons

#### Règle 1.1.1 : Doublons Stricts

- **Méthode** : Vérification sur l'ensemble des colonnes
- **Action** : Suppression automatique
- **Justification** : Les doublons exacts n'apportent aucune information nouvelle

```python
# Implémentation
df = df.drop_duplicates(keep='first')
```

#### Règle 1.1.2 : Doublons Partiels

- **Méthode** : Comparaison sur colonnes cliniquement pertinentes :
  - `patient_id`
  - `age`
  - `pathology`
  - `assessment` (BI-RADS)
  - Caractéristiques morphologiques (`margin`, `shape`, `density`)
- **Action** :
  - Si toutes ces colonnes sont identiques → doublon probable
  - Vérification manuelle des autres colonnes
  - Conservation de l'enregistrement le plus complet

**Justification** : Éviter de perdre des cas uniques tout en éliminant les vrais doublons

---

### 1.2. Gestion des Valeurs Manquantes

#### Règle 1.2.1 : Colonnes Critiques (TOLÉRANCE ZÉRO)

**Colonnes ne pouvant être manquantes** :

- `pathology` : Diagnostic final (benign/malignant)
- `assessment` : Score BI-RADS
- `margin` : Caractéristique des bords
- `shape` : Forme de la lésion
- `density` : Densité mammaire

**Action** :

- Si valeur manquante → suppression de la ligne
- **Justification médicale** : Ces informations sont essentielles pour le diagnostic

#### Règle 1.2.2 : Colonnes Non-Critiques

**Colonnes tolérantes** :

- `patient_id` : Si manquant, générer un ID unique
- `image_id` : Si manquant, peut être recréé
- `subtlety` : Peut être imputé par la médiane
- Notes/commentaires : Remplir par "N/A"

**Action** :

- Imputation ou remplissage selon le type
- **Justification** : Ne pas perdre de cas pour des métadonnées

---

## 2. COHÉRENCE SÉMANTIQUE ET MÉDICALE

### 2.1. Normalisation du Texte

#### Règle 2.1.1 : Standardisation de Base

1. **Conversion en minuscules** (sauf acronymes médicaux)
2. **Suppression des espaces multiples**
3. **Suppression des caractères spéciaux non médicaux**
4. **Suppression des accents** (pour harmonisation)

```python
# Exemple
texte = texte.lower().strip()
texte = re.sub(r'\s+', ' ', texte)
texte = unidecode(texte)
```

#### Règle 2.1.2 : Harmonisation des Libellés

**Pathology** :

- `"benign"`, `"Benign"`, `"BENIGN"`, `"bénin"` → **`"benign"`**
- `"malignant"`, `"malign"`, `"malig."`, `"cancer"` → **`"malignant"`**

**Margin** :

- `"circumscribed"`, `"circums."`, `"well-defined"` → **`"circumscribed"`**
- `"ill-defined"`, `"ill defined"`, `"poorly defined"` → **`"ill-defined"`**
- `"spiculated"`, `"spic."`, `"spiky"` → **`"spiculated"`**
- `"microlobulated"`, `"micro-lob"` → **`"microlobulated"`**

**Shape** :

- `"round"`, `"circular"` → **`"round"`**
- `"oval"`, `"ovale"` → **`"oval"`**
- `"lobulated"`, `"lobular"` → **`"lobulated"`**
- `"irregular"`, `"irreg"` → **`"irregular"`**

**Density** :

- `"A"`, `"a"`, `"type a"` → **`"A"`** (presque entièrement graisseuse)
- `"B"`, `"b"`, `"type b"` → **`"B"`** (densités fibroglandulaires dispersées)
- `"C"`, `"c"`, `"type c"` → **`"C"`** (tissu dense hétérogène)
- `"D"`, `"d"`, `"type d"` → **`"D"`** (tissu extrêmement dense)

**Justification** : Assurer la cohérence et éviter les duplicata de catégories

---

### 2.2. Respect des Règles Médicales BI-RADS

#### Règle 2.2.1 : Cohérence BI-RADS ↔ Pathology

**Matrice de cohérence attendue** :

| BI-RADS | Signification      | Pathology Attendue | Probabilité Malignité |
| ------- | ------------------ | ------------------ | --------------------- |
| 1       | Négatif            | Benign             | <2%                   |
| 2       | Bénin              | Benign             | <2%                   |
| 3       | Probablement bénin | Benign (>90%)      | <10%                  |
| 4       | Anomalie suspecte  | Malignant (20-90%) | 20-90%                |
| 5       | Hautement suspect  | Malignant (>90%)   | >95%                  |

**Règles de validation** :

1. **BI-RADS 1-2 + Malignant** → ⚠️ **ANOMALIE CRITIQUE**

   - Action : Marquer comme incohérence
   - Possibilités : Erreur de saisie OU cas très rare
   - Nécessite révision manuelle

2. **BI-RADS 5 + Benign** → ⚠️ **ANOMALIE MODÉRÉE**

   - Action : Vérifier les autres caractéristiques
   - Possibilités : Faux positif radiologique OR erreur
   - Peut être conservé si justifié

3. **BI-RADS 3-4** → Tolérance à la variabilité
   - Ces catégories sont ambiguës par nature

**Implémentation** :

```python
def check_birads_pathology_consistency(row):
    birads = row['assessment']
    pathology = row['pathology']

    # Incohérences critiques
    if birads in [1, 2] and pathology == 'malignant':
        return 'CRITICAL_INCONSISTENCY'
    if birads == 5 and pathology == 'benign':
        return 'MODERATE_INCONSISTENCY'

    return 'CONSISTENT'

df['consistency_check'] = df.apply(check_birads_pathology_consistency, axis=1)
```

#### Règle 2.2.2 : Cohérence Morphologique

**Caractéristiques fortement associées au cancer** :

| Caractéristique               | Type   | Association Malignité | Action si incohérent     |
| ----------------------------- | ------ | --------------------- | ------------------------ |
| `margin = "spiculated"`       | Margin | ÉLEVÉE (>80%)         | Flag si benign           |
| `shape = "irregular"`         | Shape  | MODÉRÉE (>50%)        | Vérifier autres critères |
| `margin = "circumscribed"`    | Margin | FAIBLE (<10%)         | Flag si malignant        |
| `shape = "round"` ou `"oval"` | Shape  | FAIBLE (<15%)         | Flag si malignant        |

**Matrice de décision** :

```
SI margin="spiculated" ET pathology="benign"
   → Marquer comme "RARE_CASE" (peut être vrai mais rare)
   → Conserver mais annoter

SI margin="circumscribed" ET shape="round" ET pathology="malignant"
   → Marquer comme "ATYPICAL_MALIGNANT"
   → Vérifier BI-RADS (devrait être bas)
```

---

### 2.3. Harmonisation des Types de Données

#### Règle 2.3.1 : Conversion Automatique

| Colonne      | Type Original | Type Cible | Justification         |
| ------------ | ------------- | ---------- | --------------------- |
| `patient_id` | object/int    | int64      | Identifiant numérique |
| `age`        | object/int    | int64      | Valeur numérique      |
| `assessment` | object/int    | int64      | Score 1-5             |
| `pathology`  | object        | category   | Variable catégorielle |
| `margin`     | object        | category   | Variable catégorielle |
| `shape`      | object        | category   | Variable catégorielle |
| `density`    | object        | category   | Variable catégorielle |
| `subtlety`   | object/int    | int64      | Score 1-5             |

**Implémentation** :

```python
# Conversions
df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['assessment'] = df['assessment'].astype(int)

# Catégories
categorical_cols = ['pathology', 'margin', 'shape', 'density']
for col in categorical_cols:
    df[col] = df[col].astype('category')
```

---

## 3. QUALITÉ STATISTIQUE

### 3.1. Détection des Outliers

#### Règle 3.1.1 : Outliers sur l'Âge

**Méthode IQR (Interquartile Range)** :

```python
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['age'] < lower_bound) | (df['age'] > upper_bound)]
```

**Règles de décision** :

| Âge       | Décision     | Justification                       |
| --------- | ------------ | ----------------------------------- |
| < 18 ans  | ⚠️ Vérifier  | Cancer du sein rare chez les jeunes |
| 18-25 ans | ⚠️ Annoter   | Possible mais rare                  |
| 25-90 ans | ✅ Conserver | Plage normale                       |
| > 90 ans  | ⚠️ Vérifier  | Possible mais vérifier la saisie    |
| > 120 ans | ❌ Supprimer | Biologiquement impossible           |

#### Règle 3.1.2 : Outliers sur Taille/Diamètre (si disponible)

- **Valeurs < 0** → Erreur de saisie, suppression
- **Valeurs > 200 mm** → Vérification (très rare)
- **Valeurs aberrantes** détectées par IQR → Annotation

---

### 3.2. Vérification des Distributions

#### Règle 3.2.1 : Valeurs Rares

**Critère** : Catégories représentant < 1% des données

**Action** :

1. Identifier les catégories rares
2. Décider selon pertinence clinique :
   - **Si cliniquement pertinent** → Conserver et annoter "RARE_CATEGORY"
   - **Si non pertinent** → Fusionner dans "AUTRE" ou supprimer

**Exemple** :

```python
# Distribution des margins
margin_dist = df['margin'].value_counts(normalize=True) * 100

rare_margins = margin_dist[margin_dist < 1.0].index.tolist()
print(f"Catégories rares (< 1%): {rare_margins}")

# Décision selon contexte médical
# Ex: "spiculated" rare mais TRÈS pertinent → conserver
```

---

## 4. PRÉPARATION POUR MODÉLISATION

### 4.1. Encodage des Variables

#### Règle 4.1.1 : Variable Cible (Pathology)

**Méthode** : LabelEncoder

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['pathology_encoded'] = le.fit_transform(df['pathology'])

# Mapping : {'benign': 0, 'malignant': 1}
```

#### Règle 4.1.2 : Variables Catégorielles (Features)

**Méthode** : OneHotEncoder pour `margin`, `shape`, `density`

```python
from sklearn.preprocessing import OneHotEncoder

cat_features = ['margin', 'shape', 'density']
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

encoded_features = ohe.fit_transform(df[cat_features])
feature_names = ohe.get_feature_names_out(cat_features)

df_encoded = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
df_final = pd.concat([df, df_encoded], axis=1)
```

---

### 4.2. Standardisation des Variables Numériques

#### Règle 4.2.1 : StandardScaler

**Colonnes concernées** : `age`, `subtlety`, diamètre (si disponible)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = ['age', 'subtlety']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

**Justification** : Mise à l'échelle pour les algorithmes sensibles (SVM, KNN, réseaux de neurones)

---

### 4.3. Documentation des Transformations

#### Règle 4.3.1 : Logging Systématique

**Chaque transformation doit être documentée** :

```python
transformation_log = {
    'timestamp': datetime.now().isoformat(),
    'transformations': [
        {
            'step': 1,
            'operation': 'Remove duplicates',
            'rows_before': len(df_before),
            'rows_after': len(df_after),
            'rows_removed': len(df_before) - len(df_after),
            'justification': 'Doublons exacts détectés'
        },
        {
            'step': 2,
            'operation': 'Normalize text (pathology)',
            'affected_column': 'pathology',
            'unique_values_before': ['Benign', 'benign', 'MALIGNANT', 'malignant'],
            'unique_values_after': ['benign', 'malignant'],
            'justification': 'Harmonisation des libellés'
        },
        # ... autres transformations
    ]
}

# Sauvegarder
with open('cleaning_log.json', 'w') as f:
    json.dump(transformation_log, f, indent=4)
```

---

## 5. VALIDATION MÉDICALE FINALE

### 5.1. Checklist de Validation

Avant de considérer le dataset comme nettoyé :

- [ ] **Cohérence BI-RADS ↔ Pathology** : < 5% d'incohérences
- [ ] **Outliers d'âge** : Justifiés ou supprimés
- [ ] **Caractéristiques morphologiques** : Cohérentes avec pathology
- [ ] **Distributions** : Plausibles médicalement
- [ ] **Encodage** : Correct et documenté
- [ ] **Logs** : Complets et traçables

### 5.2. Métriques de Qualité Post-Nettoyage

| Métrique          | Valeur Cible | Description                                      |
| ----------------- | ------------ | ------------------------------------------------ |
| Complétude        | > 98%        | % de valeurs non-manquantes (colonnes critiques) |
| Cohérence BI-RADS | > 90%        | % de cas cohérents entre BI-RADS et Pathology    |
| Doublons          | 0%           | Aucun doublon strict                             |
| Outliers gérés    | 100%         | Tous outliers vérifiés/annotés                   |
| Documentation     | 100%         | Toutes transformations documentées               |

---

## 📖 RÉFÉRENCES MÉDICALES

1. **BI-RADS Atlas** (American College of Radiology)
2. **D'Orsi CJ et al.** - ACR BI-RADS® Atlas, Breast Imaging Reporting and Data System
3. **Sickles EA et al.** - ACR BI-RADS® Mammography (2013)

---

**Document créé le** : 01 Décembre 2025  
**Dernière mise à jour** : 01 Décembre 2025  
**Version** : 1.0
