import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer 

# 1. LOAD AND PREPARE DATA
# ---------------------------------------------------------
# Load the dataset
try:
    df = pd.read_csv('preeclampsia.csv', header=None)
except FileNotFoundError:
    print("Error: 'preeclampsia.csv' not found. Please check the file path.")
    exit()

#the dataset columns are wrong so we switch those up
df = df.rename(columns={'sysbp': 'diabp', 'diabp': 'sysbp'})

#standardize columns 
df.columns = df.columns.str.strip().str.lower().str.replace(":", "", regex=False)

# Assign column names based on dataset structure
expected_columns = ['age', 'gest_age', 'height', 'weight', 'bmi', 'sysbp', 'diabp', 'hb', 
           'pcv', 'tsh', 'platelet', 'creatinine', 'plgfsflt', 'seng', 'cysc', 
           'pp_13', 'glycerides', 'htn', 'diabetes', 'fam_htn', 'sp_art', 
           'occupation', 'diet', 'activity', 'sleep']

#check if any columns are missing( error handling)
missing = set(expected_columns) - set(df.columns)
if missing:
    raise ValueError(f" Missing required columns: {missing}")

# Convert all columns to numeric (handle any parsing issues)
numeric_columns = ['age', 'gest_age', 'height', 'weight', 'bmi', 'sysbp', 'diabp', 
                   'hb', 'pcv', 'tsh', 'platelet', 'creatinine', 'plgfsflt', 
                   'seng', 'cysc', 'pp_13', 'glycerides']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Handle binary columns safely
binary_cols = ['htn', 'diabetes', 'fam_htn', 'sp_art']
df[binary_cols] = df[binary_cols].apply(pd.to_numeric, errors='coerce')

# 2. unsupervised risk phenotype discovery
cluster_cols = [
    'sysbp',
    'diabp',
    'plgfsflt',
    'creatinine',
    'seng',
    'cysc',
    'pp_13'
]

X_cluster = df[cluster_cols]

# Impute + scale
cluster_imputer = SimpleImputer(strategy='median')
X_cluster_imp = cluster_imputer.fit_transform(X_cluster)

cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster_imp)

# KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=100)
df['cluster_label'] = kmeans.fit_predict(X_cluster_scaled)

sil = silhouette_score(X_cluster_scaled, df['cluster_label'])
print(f"Silhouette Score: {sil:.3f}")

# 3. IDENTIFY HIGH-RISK PHENOTYPE (CLINICAL WEIGHTING)
cluster_centers = kmeans.cluster_centers_

feature_weights = np.array([
    1.0,  # sysbp
    1.0,  # diabp
    1.3,  # plgfsflt (key placental biomarker)
    1.1,  # creatinine
    1.0,  # seng
    1.0,  # cysc
    1.0   # pp_13
])

risk_scores = np.dot(cluster_centers, feature_weights)
high_risk_cluster = np.argmax(risk_scores)

df['phenotype_group'] = (df['cluster_label'] == high_risk_cluster).astype(int)

print(f"\nHigh-Risk Cluster Identified: Cluster {high_risk_cluster}")

# Distribution check
counts = df['phenotype_group'].value_counts(normalize=True) * 100
print("\nPhenotype Distribution (%):")
print(counts)


# 4. MEDICAL PLAUSIBILITY CHECK
print("\n--- Medical Sanity Check ---")
print(df.groupby('phenotype_group')[['sysbp','diabp','plgfsflt','creatinine']].mean())


# 5. SUPERVISED PHENOTYPE CLASSIFIER
print("\n--- Training Phenotype Classifier ---")

X = df[numeric_columns + binary_cols]
y = df['phenotype_group']

clf_imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(
    clf_imputer.fit_transform(X),
    columns=X.columns
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)

# =========================================================
# 6. EVALUATION
# =========================================================
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print("\nAccuracy:", rf.score(X_test, y_test))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
fi = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Risk Drivers:")
print(fi.head(10))

# =========================================================
# 7. VISUALIZATIONS
# =========================================================
fig, ax = plt.subplots(1, 2, figsize=(14,5))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt="d", cmap="Blues", ax=ax[0]
)
ax[0].set_title("Confusion Matrix")

sns.barplot(
    data=fi.head(10),
    x="importance",
    y="feature",
    ax=ax[1]
)
ax[1].set_title("Top Biomarkers Driving Risk")

plt.tight_layout()
plt.show()

# =========================================================
# 8. SAVE ARTIFACTS
# =========================================================
joblib.dump(rf, "preeclampsia_phenotype_model.pkl")
joblib.dump(cluster_imputer, "cluster_imputer.pkl")
joblib.dump(cluster_scaler, "cluster_scaler.pkl")
joblib.dump(clf_imputer, "classifier_imputer.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\n Pipeline Complete")
print("Flow: Unsupervised Phenotyping → Clinical Validation → Rapid Risk Prediction")
