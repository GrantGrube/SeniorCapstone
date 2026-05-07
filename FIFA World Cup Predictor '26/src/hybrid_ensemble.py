import pandas as pd
import numpy as np
import math
from collections import defaultdict
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import poisson as sci_poisson
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Config
data_path = "FIFA World Cup Predictor '26/Data/Processed/training_dataset.csv"
train_cutoff = 2022
xgb_blend = 0.30  # 0 = pure Poisson, 1 = pure XGBoost classifier

league_avg_goals = 1.35 # expected goals for each team
labels = ["Home Win", "Draw", "Away Win"]

# Separate draw thresholds per model
draw_threshold_xgb = 0.42 # XGBoost draw threshold adjuster
draw_threshold_poisson = 0.28 # Poisson draw threshold adjuster
draw_threshold_hybrid = 0.32 # blended middle ground

# Dixon-Coles tau correction strength 
# 0 = off, 1 = full correction.
DC_TAU = 0.15

# Load & Sort Data
print("Loading data...")
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df["year"] = df["date"].dt.year
df = df.dropna(subset=["home_elo", "away_elo"]).reset_index(drop=True)

# Stage Flags
for col in ["stage_Group Stage", "stage_Knockout", "is_knockout"]:
    if col not in df.columns:
        df[col] = 0
df["stage_Group Stage"] = df["stage_Group Stage"].astype(int)
df["stage_Knockout"] = df["stage_Knockout"].astype(int)
df["is_knockout"] = df["is_knockout"].astype(int)
df["stage_pressure"] = df["is_knockout"] * 1.5 + df["stage_Knockout"] * 1.2

# Rolling Team Stats
team_games = defaultdict(int)
team_wins = defaultdict(int)
team_draws = defaultdict(int)
team_scored = defaultdict(int)
team_conceded = defaultdict(int)
team_history = defaultdict(list)

home_win_rate, away_win_rate = [], []
home_draw_rate, away_draw_rate = [], []
home_attack, away_attack = [], []
home_defense, away_defense = [], []
home_exp, away_exp = [], []
home_form, away_form = [], []
home_momentum, away_momentum = [], []

def safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0

for _, row in df.iterrows():
    h, a = row["home_team"], row["away_team"]
    hg, ag = team_games[h], team_games[a]

    home_win_rate.append(team_wins[h] / hg if hg else 0)
    away_win_rate.append(team_wins[a] / ag if ag else 0)
    home_draw_rate.append(team_draws[h] / hg if hg else 0)
    away_draw_rate.append(team_draws[a] / ag if ag else 0)
    home_attack.append(team_scored[h] / hg if hg else 0)
    away_attack.append(team_scored[a] / ag if ag else 0)
    home_defense.append(team_conceded[h] / hg if hg else 0)
    away_defense.append(team_conceded[a] / ag if ag else 0)
    home_exp.append(hg)
    away_exp.append(ag)

    h_hist = team_history[h]
    a_hist = team_history[a]
    home_form.append(safe_mean([g[0] - g[1] for g in h_hist[-5:]]))
    away_form.append(safe_mean([g[0] - g[1] for g in a_hist[-5:]]))
    home_momentum.append(safe_mean([g[0] - g[1] for g in h_hist[-10:]]))
    away_momentum.append(safe_mean([g[0] - g[1] for g in a_hist[-10:]]))

    # Update AFTER reading (no leakage)
    team_games[h] += 1; team_games[a] += 1
    team_scored[h] += row["home_score"]; team_scored[a] += row["away_score"]
    team_conceded[h] += row["away_score"];  team_conceded[a] += row["home_score"]
    if   row["home_score"] > row["away_score"]:
        team_wins[h] += 1
    elif row["away_score"] > row["home_score"]:
        team_wins[a] += 1
    else:
        team_draws[h] += 1; team_draws[a] += 1
    team_history[h].append((row["home_score"], row["away_score"]))
    team_history[a].append((row["away_score"], row["home_score"]))

# Match Context
df["home_win_rate"] = home_win_rate
df["away_win_rate"] = away_win_rate
df["home_draw_rate"] = home_draw_rate
df["away_draw_rate"] = away_draw_rate
df["home_attack"] = home_attack
df["away_attack"] = away_attack
df["home_defense"] = home_defense
df["away_defense"] = away_defense
df["home_experience"] = home_exp
df["away_experience"] = away_exp
df["home_form"] = home_form
df["away_form"] = away_form
df["home_momentum"] = home_momentum
df["away_momentum"] = away_momentum

# Team Strength Features
df["elo_diff"] = df["home_elo"] - df["away_elo"]
df["attack_diff"] = df["home_attack"] - df["away_attack"]
df["defense_diff"] = df["away_defense"] - df["home_defense"]
df["win_rate_diff"] = df["home_win_rate"] - df["away_win_rate"]
df["experience_ratio"] = np.log((df["home_experience"] + 1) / (df["away_experience"] + 1))
df["form_diff"] = df["home_form"] - df["away_form"]
df["momentum_diff"] = df["home_momentum"] - df["away_momentum"]
df["volatility"] = df[["home_attack", "away_attack"]].std(axis=1)

# Draw signal features
df["strength_balance"] = 1 / (1 + abs(df["elo_diff"]))
df["form_balance"] = 1 / (1 + abs(df["form_diff"]))
df["attack_balance"] = 1 / (1 + abs(df["attack_diff"]))
df["defense_balance"] = 1 / (1 + abs(df["defense_diff"]))
df["draw_pressure"] = (df["strength_balance"] + df["form_balance"] +
                          df["attack_balance"]   + df["defense_balance"])
df["draw_affinity"] = (df["home_draw_rate"] + df["away_draw_rate"]) / 2

# Features & Target
features = [
    "attack_diff", "defense_diff", "win_rate_diff", "experience_ratio",
    "elo_diff", "form_diff", "momentum_diff", "volatility", "stage_pressure",
    "strength_balance", "form_balance", "attack_balance", "defense_balance",
    "draw_pressure", "draw_affinity", "home_draw_rate", "away_draw_rate",
]

result_map = {"Away Win": 0, "Draw": 1, "Home Win": 2}
df["y"] = df["result"].map(result_map)

# Class Weights
class_counts = df["y"].value_counts().sort_index()
total = class_counts.sum()
base_weights = total / (3 * class_counts)
DRAW_EXTRA = 1.0   # extra boost to fix draw predictions

# TRAIN / TEST SPLIT
train_df = df[df["year"] < train_cutoff].dropna(subset=features + ["y"]).copy()
test_df  = df[df["year"] >= train_cutoff].dropna(subset=features + ["y"]).copy()

X_train, y_train = train_df[features], train_df["y"]
X_test,  y_test  = test_df[features],  test_df["y"]

w_map = {
    0: float(base_weights[0]),
    1: float(base_weights[1]) * DRAW_EXTRA,
    2: float(base_weights[2]),
}
sample_weight = y_train.map(w_map).values

print(f"Train: {len(train_df):,} matches  |  Test: {len(test_df):,} matches")
print(f"Class weights → Away:{w_map[0]:.2f}  Draw:{w_map[1]:.2f}  Home:{w_map[2]:.2f}\n")

# Cross Validation of Model
print("Running cross-validation...")
tscv = TimeSeriesSplit(n_splits=5)
cv_model = XGBClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=6,
    subsample=0.85, colsample_bytree=0.75,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", random_state=42,
)

cv_scores = []
for train_idx, val_idx in tscv.split(X_train):
    X_cv_train = X_train.iloc[train_idx]
    X_cv_val = X_train.iloc[val_idx]
    y_cv_train = y_train.iloc[train_idx]
    y_cv_val = y_train.iloc[val_idx]
    sw_cv = sample_weight[train_idx]

    cv_model.fit(X_cv_train, y_cv_train, sample_weight=sw_cv)
    preds = cv_model.predict(X_cv_val)
    cv_scores.append(accuracy_score(y_cv_val, preds))

cv_scores = np.array(cv_scores)
print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# MODEL 1 — XGBoost Classifier  (weighted + draw-aware)
print("Training XGBoost classifier.")
clf = XGBClassifier(
    n_estimators=1800, learning_rate=0.02, max_depth=6,
    min_child_weight=3, subsample=0.85, colsample_bytree=0.75,
    gamma=0.15, reg_alpha=0.3, reg_lambda=2.5,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", random_state=42,
)
clf.fit(X_train, y_train, sample_weight=sample_weight)
print(" Classifier trained.")

# MODEL 2 — XGBoost Regressors  (goal counts)
print("Training XGBoost goal regressors.")
reg_home = XGBRegressor(
    n_estimators=600, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
reg_away = XGBRegressor(
    n_estimators=600, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
reg_home.fit(X_train, train_df["home_score"])
reg_away.fit(X_train, train_df["away_score"])
print(" Goal regressors trained.\n")

# DIXON-COLES TAU CORRECTION
# Helps model boost draws
def dc_tau(x, y, lam_h, lam_a, rho):
    if   x == 0 and y == 0: return 1 - lam_h * lam_a * rho
    elif x == 1 and y == 0: return 1 + lam_a * rho
    elif x == 0 and y == 1: return 1 + lam_h * rho
    elif x == 1 and y == 1: return 1 - rho
    else:                   return 1.0

# Estimate rho from training draw rate vs naive Poisson draw rate
train_draw_rate = (y_train == 1).mean()
mean_lam = train_df["home_attack"].mean()
naive_poi_draw = np.exp(-2 * mean_lam) * sum(
    (mean_lam ** k / math.factorial(k)) ** 2
    for k in range(6)
)
rho_est = (train_draw_rate - naive_poi_draw) / max(naive_poi_draw, 1e-6)
rho_est = float(np.clip(rho_est * DC_TAU, -0.2, 0.2))
print(f"Dixon-Coles rho estimate: {rho_est:.4f}  (train draw rate: {train_draw_rate:.3f})\n")

def poisson_outcome_probs_dc(lam_h, lam_a, max_goals=10):
    goals = np.arange(0, max_goals + 1)
    ph = sci_poisson.pmf(goals, lam_h)
    pa = sci_poisson.pmf(goals, lam_a)
    joint = np.outer(ph, pa)

    for x in range(min(2, max_goals + 1)):
        for y in range(min(2, max_goals + 1)):
            joint[x, y] *= dc_tau(x, y, lam_h, lam_a, rho_est)

    joint /= joint.sum()

    p_home = float(np.tril(joint, -1).sum())
    p_away = float(np.triu(joint,  1).sum())
    p_draw = float(np.trace(joint))
    return p_home, p_draw, p_away

# Hybrid Predictions
print("Generating predictions on test set.")

clf_probs_test = clf.predict_proba(X_test) # [away=0, draw=1, home=2]
lam_h_test = np.maximum(0.05, reg_home.predict(X_test))
lam_a_test = np.maximum(0.05, reg_away.predict(X_test))

y_pred_hybrid = []
y_pred_clf_only = []
y_pred_poi_only = []

for i in range(len(test_df)):
    # Classifier probs (away=0, draw=1, home=2)
    p_away_c, p_draw_c, p_home_c = clf_probs_test[i]

    # Dixon-Coles Poisson probs
    p_home_p, p_draw_p, p_away_p = poisson_outcome_probs_dc(lam_h_test[i], lam_a_test[i])

    # Hybrid 
    p_home_b = xgb_blend * p_home_c + (1 - xgb_blend) * p_home_p
    p_draw_b = xgb_blend * p_draw_c + (1 - xgb_blend) * p_draw_p
    p_away_b = xgb_blend * p_away_c + (1 - xgb_blend) * p_away_p

    tot = p_home_b + p_draw_b + p_away_b
    p_home_b /= tot; p_draw_b /= tot; p_away_b /= tot

    if p_draw_b >= draw_threshold_hybrid:
        y_pred_hybrid.append(1)
    else:
        y_pred_hybrid.append(int(np.argmax([p_away_b, p_draw_b, p_home_b])))

    # Classifier-only baseline
    tot_c = p_home_c + p_draw_c + p_away_c
    p_home_c /= tot_c; p_draw_c /= tot_c; p_away_c /= tot_c
    if p_draw_c >= draw_threshold_xgb:
        y_pred_clf_only.append(1)
    else:
        y_pred_clf_only.append(int(np.argmax([p_away_c, p_draw_c, p_home_c])))

    # Poisson-only baseline 
    if p_draw_p >= draw_threshold_poisson:
        y_pred_poi_only.append(1)
    else:
        y_pred_poi_only.append(int(np.argmax([p_away_p, p_draw_p, p_home_p])))

print(" Predictions ready.\n")

# Evaluation
models_eval = [
    ("XGBoost Classifier  (weighted + draw threshold)", y_pred_clf_only),
    ("Poisson DC-corrected  (draw threshold)",          y_pred_poi_only),
    (f"Hybrid blend={xgb_blend}  (weighted + DC + threshold)", y_pred_hybrid),
]

for name, preds in models_eval:
    print("=" * 70)
    print(f" {name.upper()}")
    print("=" * 70)
    print(f"  Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Away Win", "Draw", "Home Win"], zero_division=0))

# Normalized Confusion Matrices
os.makedirs("output", exist_ok=True)

fig = plt.figure(figsize=(19, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

model_names = [
    f"XGBoost\n(weighted + draw threshold={draw_threshold_xgb})",
    f"Poisson DC-corrected\n(draw threshold={draw_threshold_poisson})",
    f"Hybrid  (XGB {int(xgb_blend*100)}% / Poisson {int((1-xgb_blend)*100)}%)\n"
    f"weighted + DC + threshold={draw_threshold_hybrid}",
]
all_preds = [y_pred_clf_only, y_pred_poi_only, y_pred_hybrid]
label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
y_test_labels = [label_map[p] for p in y_test]

for col, (name, preds) in enumerate(zip(model_names, all_preds)):
    pred_labels = [label_map[p] for p in preds]
    cm = confusion_matrix(y_test_labels, pred_labels, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    ax = fig.add_subplot(gs[col])
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.80, format="{x:.0%}")

    ax.set_xticks(range(3)); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)

    acc = accuracy_score(y_test, preds)
    ax.set_title(f"{name}\nAccuracy: {acc:.3f}", fontsize=9, pad=10)

    thresh = 0.5
    for i in range(3):
        for j in range(3):
            pct = cm_norm[i, j]
            ax.text(j, i, f"{pct*100:.1f}%",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if pct > thresh else "black")

fig.suptitle(
    f"Normalized Confusion Matrices — Test set n={len(y_test)}  "
    f"(year ≥ {train_cutoff})  |  "
    f"Thresholds — CLF:{draw_threshold_xgb}  POI:{draw_threshold_poisson}  HYB:{draw_threshold_hybrid}  |  "
    f"Dixon-Coles rho: {rho_est:.3f}  |  Draw class weight ×{DRAW_EXTRA}",
    fontsize=9, y=1.03,
)
plt.savefig("output/confusion_matrices_improved.png", dpi=150, bbox_inches="tight")
plt.show()
print("Confusion matrices → output/confusion_matrices_improved.png\n")

# Feature Importance
importance_df = pd.DataFrame({
    "feature":    features,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)
importance_df["rank"] = range(1, len(importance_df) + 1)

print("=" * 55)
print(" FEATURE IMPORTANCE (XGBoost Classifier)")
print("=" * 55)
print(importance_df.to_string(index=False))
