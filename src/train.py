import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/iris.csv")

# Features and target
X = data.drop("species", axis=1)
y = data["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Model")
print("-------------------")
print("Number of trees:", 100)
print("Maximum depth:", 5)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create models directory
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/random_forest_v1.pkl")

print("\nModel saved to:")
print("models/random_forest_v1.pkl")