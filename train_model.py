import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("hand_gesture_data.csv")

print("CSV Columns:", data.columns)

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("✅ Model Accuracy:", accuracy)

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as gesture_model.pkl")
