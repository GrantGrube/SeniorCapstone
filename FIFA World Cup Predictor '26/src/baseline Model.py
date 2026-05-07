import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

# Load Data
df = pd.read_csv("FIFA World Cup Predictor '26/Data/Processed/training_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df["year"] = df["date"].dt.year

# Cumulative Team Stats (computed before each match — no leakage)
team_games = defaultdict(int)
team_wins = defaultdict(int)
team_draws = defaultdict(int)
team_goals_scored = defaultdict(int)
team_goals_conceded = defaultdict(int)
team_goal_diff = defaultdict(int)
team_low_scoring = defaultdict(int) # games where team scored <= 1

home_win_rate_list = []
away_win_rate_list = []
home_goal_diff_avg_list = []
away_goal_diff_avg_list = []
home_attack_list = []
away_attack_list = []
home_defense_list = []
away_defense_list = []
home_experience_list = []
away_experience_list = []

# Draw tendency lists
home_draw_rate_list = []
away_draw_rate_list = []
home_draw_tendency_list = []
away_draw_tendency_list = []

for idx, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    hg = team_games[home]
    ag = team_games[away]

    home_win_rate_list.append(team_wins[home] / hg if hg > 0 else 0)
    away_win_rate_list.append(team_wins[away] / ag if ag > 0 else 0)

    home_goal_diff_avg_list.append(team_goal_diff[home] / hg if hg > 0 else 0)
    away_goal_diff_avg_list.append(team_goal_diff[away] / ag if ag > 0 else 0)

    home_attack_list.append(team_goals_scored[home] / hg if hg > 0 else 0)
    away_attack_list.append(team_goals_scored[away] / ag if ag > 0 else 0)

    home_defense_list.append(team_goals_conceded[home] / hg if hg > 0 else 0)
    away_defense_list.append(team_goals_conceded[away] / ag if ag > 0 else 0)

    home_experience_list.append(hg)
    away_experience_list.append(ag)

    # Draw tendency, captured before update
    home_draw_rate_list.append(team_draws[home] / hg if hg > 0 else 0)
    away_draw_rate_list.append(team_draws[away] / ag if ag > 0 else 0)
    home_draw_tendency_list.append(team_low_scoring[home] / hg if hg > 0 else 0)
    away_draw_tendency_list.append(team_low_scoring[away] / ag if ag > 0 else 0)

    # Update AFTER reading (no leakage)
    home_goals = row["home_score"]
    away_goals = row["away_score"]

    team_games[home] += 1
    team_games[away] += 1
    team_goals_scored[home] += home_goals
    team_goals_scored[away] += away_goals
    team_goals_conceded[home] += away_goals
    team_goals_conceded[away] += home_goals
    team_goal_diff[home] += home_goals - away_goals
    team_goal_diff[away] += away_goals - home_goals

    if home_goals > away_goals:
        team_wins[home] += 1
    elif away_goals > home_goals:
        team_wins[away] += 1
    else:
        team_draws[home] += 1
        team_draws[away] += 1

    if home_goals <= 1:
        team_low_scoring[home] += 1
    if away_goals <= 1:
        team_low_scoring[away] += 1

df["home_win_rate"] = home_win_rate_list
df["away_win_rate"] = away_win_rate_list
df["home_goal_diff_avg"] = home_goal_diff_avg_list
df["away_goal_diff_avg"] = away_goal_diff_avg_list
df["home_attack"] = home_attack_list
df["away_attack"] = away_attack_list
df["home_defense"] = home_defense_list
df["away_defense"] = away_defense_list
df["home_experience"] = home_experience_list
df["away_experience"] = away_experience_list
df["home_draw_rate"] = home_draw_rate_list
df["away_draw_rate"] = away_draw_rate_list
df["home_draw_tendency"] = home_draw_tendency_list
df["away_draw_tendency"] = away_draw_tendency_list

# Recency Weighted Features
team_history = defaultdict(list)

home_recent_form_list = []
away_recent_form_list = []
home_recent_attack_list = []
away_recent_attack_list = []
home_recent_defense_list = []
away_recent_defense_list = []
home_sparse_list = []
away_sparse_list = []

DECAY = 0.8
WINDOW = 10
MIN_HISTORY = 3


def weighted_average(values, decay):
    if len(values) == 0:
        return 0.0
    n = len(values)
    weights = np.array([np.exp(-decay * i) for i in range(n - 1, -1, -1)])
    values  = np.array(values, dtype=float)
    return float(np.sum(weights * values) / np.sum(weights))


for idx, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    home_hist = team_history[home][-WINDOW:]
    away_hist = team_history[away][-WINDOW:]

    # Sparse flag captured BEFORE updating history
    home_sparse_list.append(len(team_history[home]) < MIN_HISTORY)
    away_sparse_list.append(len(team_history[away]) < MIN_HISTORY)

    home_scored = [g[0] for g in home_hist]
    home_conceded = [g[1] for g in home_hist]
    away_scored = [g[0] for g in away_hist]
    away_conceded = [g[1] for g in away_hist]
    home_form = [g[0] - g[1] for g in home_hist]
    away_form = [g[0] - g[1] for g in away_hist]

    home_recent_form_list.append(weighted_average(home_form, DECAY))
    away_recent_form_list.append(weighted_average(away_form, DECAY))
    home_recent_attack_list.append(weighted_average(home_scored, DECAY))
    away_recent_attack_list.append(weighted_average(away_scored, DECAY))
    home_recent_defense_list.append(weighted_average(home_conceded, DECAY))
    away_recent_defense_list.append(weighted_average(away_conceded, DECAY))

    home_goals = row["home_score"]
    away_goals = row["away_score"]
    team_history[home].append((home_goals, away_goals))
    team_history[away].append((away_goals, home_goals))

df["home_recent_form"] = home_recent_form_list
df["away_recent_form"] = away_recent_form_list
df["home_recent_attack"] = home_recent_attack_list
df["away_recent_attack"] = away_recent_attack_list
df["home_recent_defense"] = home_recent_defense_list
df["away_recent_defense"] = away_recent_defense_list
df["home_data_sparse"] = home_sparse_list
df["away_data_sparse"] = away_sparse_list

# ELO Ratings
elo = defaultdict(lambda: 1500)
K = 30

home_elo_list = []
away_elo_list = []
elo_diff_list = []

for idx, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    home_elo = elo[home]
    away_elo = elo[away]

    home_elo_list.append(home_elo)
    away_elo_list.append(away_elo)
    elo_diff_list.append(home_elo - away_elo)

    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away = 1 - expected_home

    if row["home_score"] > row["away_score"]:
        actual_home, actual_away = 1, 0
    elif row["home_score"] < row["away_score"]:
        actual_home, actual_away = 0, 1
    else:
        actual_home, actual_away = 0.5, 0.5

    elo[home] += K * (actual_home - expected_home)
    elo[away] += K * (actual_away - expected_away)

df["home_elo"] = home_elo_list
df["away_elo"] = away_elo_list
df["elo_diff"] = elo_diff_list

# Feature Engineering
df["team_strength_diff"] = df["home_goal_diff_avg"] - df["away_goal_diff_avg"]
df["strength_diff_abs"] = df["team_strength_diff"].abs()
df["attack_diff"] = df["home_attack"]  - df["away_attack"]
df["defense_diff"] = df["away_defense"] - df["home_defense"] # positive = home better

df["win_rate_diff"] = df["home_win_rate"] - df["away_win_rate"]
df["win_rate_gap"] = df["win_rate_diff"].abs()

df["experience_ratio"] = np.log(
    (df["home_experience"] + 1) / (df["away_experience"] + 1)
)
df["experience_gap"] = df["experience_ratio"].abs()

df["total_goal_diff_avg"] = df["home_goal_diff_avg"] + df["away_goal_diff_avg"]
df["total_avg_goals"] = df["home_attack"] + df["away_attack"]

df["strength_parity"] = (df["strength_diff_abs"] < 0.3).astype(int)
df["close_match"] = (df["strength_diff_abs"] < 0.5).astype(int)
df["defensive_match"] = (df["total_avg_goals"] < 2.5).astype(int)

df["home_advantage"] = (~df["neutral"].astype(bool)).astype(int)

df["recent_form_diff"] = df["home_recent_form"]    - df["away_recent_form"]
df["recent_attack_diff"] = df["home_recent_attack"]  - df["away_recent_attack"]
df["recent_defense_diff"] = df["away_recent_defense"] - df["home_recent_defense"]

df["elo_diff_squared"] = df["elo_diff"] ** 2

# Draw tendency combined signals
df["draw_rate_sum"] = df["home_draw_rate"] + df["away_draw_rate"]
df["draw_tendency_sum"] = df["home_draw_tendency"] + df["away_draw_tendency"]
df["draw_affinity"] = df["draw_rate_sum"] * df["draw_tendency_sum"]

# Feature List
features = [
    "team_strength_diff",
    "strength_diff_abs",
    "attack_diff",
    "defense_diff",
    "win_rate_diff",
    "win_rate_gap",
    "experience_ratio",
    "experience_gap",
    "home_advantage",
    "strength_parity",
    "close_match",
    "defensive_match",
    "total_avg_goals",
    "total_goal_diff_avg",
    "recent_form_diff",
    "recent_attack_diff",
    "recent_defense_diff",
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_diff_squared",
    "draw_rate_sum",
    "draw_tendency_sum",
    "draw_affinity",
]

# Train / Test Split
train_all = df[df["year"] < 2022]
test = df[df["year"] == 2022]

# Exclude sparse rows from training only
train = train_all[
    (~train_all["home_data_sparse"]) &
    (~train_all["away_data_sparse"])
].copy()

X_train = train[features]
y_train = train["result"]

X_test = test[features]
y_test = test["result"]

print("Training samples (after sparse filter):", len(X_train))
print(f" Removed {len(train_all) - len(train)} sparse rows from training")
print("Testing matches (2022 WC):", len(X_test))

# Sample Weights
# Combines class balancing with a soft down-weight for any sparse test/train
# rows that survived (sparse filter only removes training rows).
base_weights = compute_sample_weight("balanced", y_train)
sample_weights = base_weights  # no sparse rows remain in train after filtering

# Find the best C value via CV
# Smaller C = stronger regularization = more evenly spread feature weights.
print("\n Regularization sweep (C values)")

param_grid = {"logreg__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}

sweep_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced"
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=False)

grid = GridSearchCV(
    sweep_pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    refit=True,
    n_jobs=-1
)
grid.fit(X_train, y_train, logreg__sample_weight=sample_weights)

best_C = grid.best_params_["logreg__C"]
print(f"Best C: {best_C}  |  Best CV accuracy: {grid.best_score_:.4f}")
print("\nFull sweep results:")
for params, mean, std in zip(
    grid.cv_results_["params"],
    grid.cv_results_["mean_test_score"],
    grid.cv_results_["std_test_score"]
):
    marker = " best" if params["logreg__C"] == best_C else ""
    print(f" C={params['logreg__C']:.3f}: {mean:.4f} ± {std:.4f}{marker}")

# Final Model; uses best C value from sweep
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        C=best_C,
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train, logreg__sample_weight=sample_weights)

# Draw Threshold Helper
# Class order from LogReg: Away Win=0, Draw=1, Home Win=2
# Overrides argmax with Draw whenever Draw probability >= threshold
def apply_draw_threshold(probs, class_order, threshold):
    draw_idx = list(class_order).index("Draw")
    predictions = []
    for row in probs:
        if row[draw_idx] >= threshold:
            predictions.append("Draw")
        else:
            predictions.append(class_order[int(np.argmax(row))])
    return predictions

# Draw threshold tuning on CV folds, never touches the test set
# Searches thresholds from 0.28 to 0.50 and picks the one with best macro F1
print("\n Draw threshold search (CV folds) ")

logreg_step = model.named_steps["logreg"]
class_order = logreg_step.classes_

best_threshold = 0.33
best_f1 = 0.0
threshold_log = []

for threshold in np.arange(0.28, 0.52, 0.02):
    fold_preds = []
    fold_labels = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_tr = X_train.iloc[train_idx]
        y_fold_tr = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        sw = compute_sample_weight("balanced", y_fold_tr)

        fold_model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                C=best_C,
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced"
            ))
        ])
        fold_model.fit(X_fold_tr, y_fold_tr, logreg__sample_weight=sw)

        probs = fold_model.predict_proba(X_fold_val)
        preds = apply_draw_threshold(probs, class_order, threshold)

        fold_preds.extend(preds)
        fold_labels.extend(y_fold_val.tolist())

    macro_f1 = f1_score(fold_labels, fold_preds, average="macro")
    threshold_log.append((round(threshold, 2), round(macro_f1, 4)))

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_threshold = round(threshold, 2)

print(f"Threshold search results:")
for t, f in threshold_log:
    marker = " <-- best" if t == best_threshold else ""
    print(f"  threshold {t:.2f} → macro F1 {f:.4f}{marker}")

print(f"\nBest draw threshold: {best_threshold}  |  macro F1: {best_f1:.4f}")

# Evaluation — default vs threshold side by side
test_probs = model.predict_proba(X_test)
y_pred_default = model.predict(X_test)
y_pred_threshold = apply_draw_threshold(test_probs, class_order, best_threshold)

print("\n" + "="*60)
print("RESULTS — DEFAULT PREDICTION (argmax)")
print("="*60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_default):.4f}")
print(f"Decay Rate: {DECAY}  |  Best C: {best_C}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default))

cm = confusion_matrix(y_test, y_pred_default, labels=class_order)
cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
print("Labeled Confusion Matrix:")
print(cm_df)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]
cm_norm_df = pd.DataFrame(cm_norm, index=class_order, columns=class_order)
print("\nNormalized Confusion Matrix:")
print(cm_norm_df.round(4))

print("\n" + "="*60)
print(f"RESULTS — WITH DRAW THRESHOLD ({best_threshold})")
print("="*60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_threshold):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_threshold))

cm_t = confusion_matrix(y_test, y_pred_threshold, labels=class_order)
cm_t_df = pd.DataFrame(cm_t, index=class_order, columns=class_order)
print("Labeled Confusion Matrix:")
print(cm_t_df)
cm_t_norm = cm_t.astype("float") / cm_t.sum(axis=1)[:, None]
cm_t_norm_df = pd.DataFrame(cm_t_norm, index=class_order, columns=class_order)
print("\nNormalized Confusion Matrix:")
print(cm_t_norm_df.round(4))

# Feature Importance
importance = np.mean(np.abs(logreg_step.coef_), axis=0)
feature_importance = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

print("\n OVERALL FEATURE IMPORTANCE ")
print(feature_importance.to_string())

print("\n FEATURE IMPORTANCE BY CLASS ")
for i, class_label in enumerate(class_order):
    print(f"\nTop features for {class_label}:")
    coefs = sorted(zip(features, logreg_step.coef_[i]), key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in coefs:
        print(f"  {feat}: {coef:.4f}")

# Sample Prediction Probabilities
prob_df = pd.DataFrame(test_probs, columns=class_order)
print("\nSample Prediction Probabilities (first 5):")
print(prob_df.head().round(4))