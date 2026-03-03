import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset

df = pd.read_csv("Data/Processed/model_training_dataset.csv")

# Ensure date is datetime
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Extract year
df["year"] = df["date"].dt.year

# Create difference features
df["avg_scored_diff"] = df["home_avg_scored"] - df["away_avg_scored"]
df["avg_conceded_diff"] = df["home_avg_conceded"] - df["away_avg_conceded"]
df["win_rate_diff"] = df["home_win_rate"] - df["away_win_rate"]
df["goal_diff_avg_diff"] = df["home_goal_diff_avg"] - df["away_goal_diff_avg"]
df["cum_wins_diff"] = df["home_cum_wins"] - df["away_cum_wins"]
df["cum_matches_diff"] = df["home_cum_matches"] - df["away_cum_matches"]
df["cum_goal_diff_diff"] = df["home_cum_goal_diff"] - df["away_cum_goal_diff"]

# Strength gap features
# Help the model detect draws
df["strength_parity"] = abs(df["goal_diff_avg_diff"])
df["win_rate_gap"] = abs(df["win_rate_diff"])
df["cum_goal_diff_diff_gap"] = abs(df["cum_goal_diff_diff"])

# Low scoring tendency Feature
# Lower total expected goals = higher draw probability
df["total_avg_goals"] = (
    df["home_avg_scored"] +
    df["home_avg_conceded"] +
    df["away_avg_scored"] +
    df["away_avg_conceded"]
) / 2

# Feature: Home Advantage Only
# If neutral == False then home advantage = 1
# If neutral == True then home advantage = 0

df["home_advantage"] = (df["neutral"] == False).astype(int)

features = [
    "avg_scored_diff",
    "avg_conceded_diff",
    "win_rate_diff",
    "goal_diff_avg_diff",
    "cum_wins_diff",
    "cum_matches_diff",
    "cum_goal_diff_diff",
    "neutral",
    "home_advantage",
    "year",
    "strength_parity",
    "win_rate_gap",
    "cum_goal_diff_diff_gap",
    "total_avg_goals"
]
X = df[features]
y = df["result"]

# Train/Test Split by Year
train = df[df["year"] < 2022]
test = df[df["year"] == 2022]

X_train = train[features]
y_train = train["result"]

X_test = test[features]
y_test = test["result"]

print("Training samples:", len(X_train))
print("Testing matches (2022):", len(X_test))

# Build Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=3000,
        solver="lbfgs"
    ))
])

# Train Model
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

logreg = model.named_steps["logreg"]

print("\n----- FEATURE COEFFICIENTS BY CLASS -----")

for i, class_label in enumerate(logreg.classes_):
    print(f"\nClass: {class_label}")
    for feature, coef in zip(features, logreg.coef_[i]):
        print(f"{feature}: {coef: .4f}")

# Prediction Probabilities
probabilities = model.predict_proba(X_test)

prob_df = pd.DataFrame(
    probabilities,
    columns=logreg.classes_
)

print("\nSample Prediction Probabilities:")
print(prob_df.head())
