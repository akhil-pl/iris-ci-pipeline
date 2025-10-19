import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Paths
data_path = "data/iris_v2.csv"
model_dir = "models"
model_path = os.path.join(model_dir, "model.joblib")

# Load data
df = pd.read_csv(data_path)
X = df.drop("species", axis=1)
y = df["species"]

# Split, train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc:.4f}")

# Save model
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_path)
print(f"📦 Model saved to {model_path}")
