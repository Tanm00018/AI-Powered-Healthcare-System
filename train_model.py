import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  # 👈 Importing evaluation metrics
from tqdm import tqdm  # 👈 for progress bar

print("🚀 Loading dataset...")

df = pd.read_csv('data/Final_Augmented_Dataset.csv')
print("✅ Dataset loaded.")

df = df.sample(n=10000, random_state=42).reset_index(drop=True)

print(f"🧾 Dataset shape: {df.shape}")
print("🧠 Columns:", df.columns.tolist())

X = df.drop('diseases', axis=1)
y = df['diseases']

print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🏋️ Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

for _ in tqdm(range(1), desc="🔄 Training Progress"):
    model.fit(X_train, y_train)

print("🎯 Model training complete.")

y_pred = model.predict(X_test) 

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open('model/disease_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model saved as disease_predictor.pkl")
