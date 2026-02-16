# ==========================================
# Imbalanced Dataset Handling Project
# Hybrid Feature Selection + Ensemble Model
# ==========================================
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
# ==========================================



# Step 1: Load Dataset
# ==========================================
data = load_breast_cancer()
X = data.data
y = data.target
print("Dataset Loaded")
print("Total Samples:", X.shape[0])
print("Total Features:", X.shape[1])
# ==========================================
# Step 2: Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# ==========================================
# Step 3: Data Scaling
# ==========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ==========================================
# Step 4: Feature Selection (RFE)
# ==========================================
rfe = RFE(LogisticRegression(max_iter=500), n_features_to_select=10)
X_train = rfe.fit_transform(X_train, y_train)
X_test = rfe.transform(X_test)
print("Selected Features:", X_train.shape[1])
# ==========================================
# Step 5: Handle Imbalance using SMOTE
# ==========================================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE balancing:")
print("Training Samples:", len(y_train))
# ==========================================
# Step 6: Train Different Models
# ==========================================
lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
svm = SVC(probability=True)
models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "SVM": svm
}
print("\nModel Comparison Results:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("-----------------------------")
# ==========================================
# Step 7: Proposed Hybrid Ensemble Model
# ==========================================
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('gb', gb)
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
print("\nProposed Hybrid Model Results:\n")
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Precision:", precision_score(y_test, y_pred_ensemble))
print("Recall:", recall_score(y_test, y_pred_ensemble))
print("F1 Score:", f1_score(y_test, y_pred_ensemble))
# ==========================================
# Step 8: Confusion Matrix Visualization
# ==========================================
cm = confusion_matrix(y_test, y_pred_ensemble)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - Proposed Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()
