import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# LOAD DATASET
# ============================================================
data = pd.read_csv("data/iris.csv")

# Features and target
X = data.drop("species", axis=1)
y = data["species"]

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# RANDOM FOREST MODEL - VERSION 2
# ============================================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# ============================================================
# TRAIN MODEL
# ============================================================
model.fit(X_train, y_train)

# ============================================================
# PREDICTION
# ============================================================
y_pred = model.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================
accuracy = accuracy_score(y_test, y_pred)

print("======================================")
print("Random Forest Model - Version 2")
print("======================================")
print("Number of trees:", 200)
print("Maximum depth:", 10)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# SAVE MODEL
# ============================================================
os.makedirs("models", exist_ok=True)

model_path = "models/random_forest_v2.pkl"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)