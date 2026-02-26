import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "model_training_dataset.csv"

df = pd.read_csv(data_path)

# Ensure date is datetime
df["date"] = pd.to_datetime(df["date"])

# Extract year
df["year"] = df["date"].dt.year

# Feature: Home Advantage Only
# If neutral == False then home advantage = 1
# If neutral == True then home advantage = 0

df["home_advantage"] = (df["neutral"] == False).astype(int)

X = df[["home_advantage"]]
y = df["result"]

# Train/Test Split by Year
train = df[df["year"] < 2022]
test = df[df["year"] == 2022]

X_train = train[["home_advantage"]]
y_train = train["result"]

X_test = test[["home_advantage"]]
y_test = test["result"]

print("Training matches:", len(train))
print("Testing matches (2022):", len(test))

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on 2022 World Cup
y_pred = model.predict(X_test)

# Evaluate Model's prediction accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy on 2022 World Cup:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
