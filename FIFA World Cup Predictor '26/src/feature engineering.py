import pandas as pd
from collections import defaultdict

# Load dataset
df = pd.read_csv("FIFA World Cup Predictor '26/Data/Processed/full_merged_dataset2.csv")

# Ensure proper date format
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# Extract year
df["year"] = df["date"].dt.year

# Stage Feature

# Match order within each World Cup
df["match_number"] = df.groupby("year").cumcount() + 1


def assign_stage_simple(row):
    year = row["year"]
    m = row["match_number"]

    # Modern World Cups (1998–2022): 64 matches
    if year >= 1998:
        return "Group Stage" if m <= 48 else "Knockout"

    # 1986–1994: 52 matches total
    elif year >= 1986:
        return "Group Stage" if m <= 36 else "Knockout"

    # 1954–1982
    elif year >= 1954:
        return "Group Stage" if m <= 24 else "Knockout"

    # 1930–1950
    else:
        return "Knockout" if m > 12 else "Group Stage"


df["stage"] = df.apply(assign_stage_simple, axis=1)

# Binary knockout indicator
df["is_knockout"] = (df["stage"] == "Knockout").astype(int)

# one-hot encoding for XGBoost
df = pd.get_dummies(df, columns=["stage"])

# Feature Engineering
team_stats = defaultdict(lambda: {
    "matches": 0,
    "goals_scored": 0,
    "goals_conceded": 0,
    "wins": 0
})

home_avg_scored = []
home_avg_conceded = []
away_avg_scored = []
away_avg_conceded = []
home_win_rate = []
away_win_rate = []
home_goal_diff_avg = []
away_goal_diff_avg = []

home_cum_wins = []
away_cum_wins = []
home_cum_matches = []
away_cum_matches = []
home_cum_goal_diff = []
away_cum_goal_diff = []

# Loop through matches in order
for _, row in df.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    home_stats = team_stats[home]
    away_stats = team_stats[away]

    home_matches = home_stats["matches"]
    away_matches = away_stats["matches"]

    # Cumulative stats (before update)
    home_cum_wins.append(home_stats["wins"])
    away_cum_wins.append(away_stats["wins"])

    home_cum_matches.append(home_matches)
    away_cum_matches.append(away_matches)

    home_cum_goal_diff.append(
        home_stats["goals_scored"] - home_stats["goals_conceded"]
    )
    away_cum_goal_diff.append(
        away_stats["goals_scored"] - away_stats["goals_conceded"]
    )

    # Rolling averages
    if home_matches > 0:
        home_avg_scored.append(home_stats["goals_scored"] / home_matches)
        home_avg_conceded.append(home_stats["goals_conceded"] / home_matches)
        home_win_rate.append(home_stats["wins"] / home_matches)
        home_goal_diff_avg.append(
            (home_stats["goals_scored"] - home_stats["goals_conceded"]) / home_matches
        )
    else:
        home_avg_scored.append(0)
        home_avg_conceded.append(0)
        home_win_rate.append(0)
        home_goal_diff_avg.append(0)

    if away_matches > 0:
        away_avg_scored.append(away_stats["goals_scored"] / away_matches)
        away_avg_conceded.append(away_stats["goals_conceded"] / away_matches)
        away_win_rate.append(away_stats["wins"] / away_matches)
        away_goal_diff_avg.append(
            (away_stats["goals_scored"] - away_stats["goals_conceded"]) / away_matches
        )
    else:
        away_avg_scored.append(0)
        away_avg_conceded.append(0)
        away_win_rate.append(0)
        away_goal_diff_avg.append(0)

    # Update stats AFTER computing features
    home_stats["matches"] += 1
    home_stats["goals_scored"] += row["home_score"]
    home_stats["goals_conceded"] += row["away_score"]

    away_stats["matches"] += 1
    away_stats["goals_scored"] += row["away_score"]
    away_stats["goals_conceded"] += row["home_score"]

    if row["home_score"] > row["away_score"]:
        home_stats["wins"] += 1
    elif row["home_score"] < row["away_score"]:
        away_stats["wins"] += 1


# Add features to dataframe

df["home_avg_scored"] = home_avg_scored
df["home_avg_conceded"] = home_avg_conceded
df["away_avg_scored"] = away_avg_scored
df["away_avg_conceded"] = away_avg_conceded
df["home_win_rate"] = home_win_rate
df["away_win_rate"] = away_win_rate
df["home_goal_diff_avg"] = home_goal_diff_avg
df["away_goal_diff_avg"] = away_goal_diff_avg
df["home_cum_wins"] = home_cum_wins
df["away_cum_wins"] = away_cum_wins
df["home_cum_matches"] = home_cum_matches
df["away_cum_matches"] = away_cum_matches
df["home_cum_goal_diff"] = home_cum_goal_diff
df["away_cum_goal_diff"] = away_cum_goal_diff
df["home_advantage"] = (df["neutral"] == False).astype(int)


# Target Variable
def match_result(row):
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    else:
        return "Draw"


df["result"] = df.apply(match_result, axis=1)
# Save Dataset
df.to_csv(
    "FIFA World Cup Predictor '26/Data/Processed/training_dataset2.csv",
    index=False
)

print("Dataset Info:")
df.info()
print(df.sample(20))
