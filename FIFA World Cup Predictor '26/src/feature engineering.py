import pandas as pd
from collections import defaultdict

# Load dataset
df = pd.read_csv("world_cup_only_dataset.csv")

# Ensure that there is a proper date format
df["date"] = pd.to_datetime(df["date"])

# Sort dates chronologically
df = df.sort_values("date").reset_index(drop=True)

# Dictionary to track team stats
team_stats = defaultdict(lambda: {
    "matches": 0,
    "goals_scored": 0,
    "goals_conceded": 0,
    "wins": 0
})

# Lists to store new features
home_avg_scored = []
home_avg_conceded = []
away_avg_scored = []
away_avg_conceded = []
home_win_rate = []
away_win_rate = []
home_goal_diff = []
away_goal_diff = []

# Loop through matches in order
for index, row in df.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    home_matches = team_stats[home]["matches"]
    away_matches = team_stats[away]["matches"]

    # Compute rolling averages before updating
    if home_matches > 0:
        home_avg_scored.append(team_stats[home]["goals_scored"] / home_matches)
        home_avg_conceded.append(team_stats[home]["goals_conceded"] / home_matches)
        home_win_rate.append(team_stats[home]["wins"] / home_matches)
        home_goal_diff.append(
            (team_stats[home]["goals_scored"] - team_stats[home]["goals_conceded"]) / home_matches
        )
    else:
        home_avg_scored.append(0)
        home_avg_conceded.append(0)
        home_win_rate.append(0)
        home_goal_diff.append(0)

    if away_matches > 0:
        away_avg_scored.append(team_stats[away]["goals_scored"] / away_matches)
        away_avg_conceded.append(team_stats[away]["goals_conceded"] / away_matches)
        away_win_rate.append(team_stats[away]["wins"] / away_matches)
        away_goal_diff.append(
            (team_stats[away]["goals_scored"] - team_stats[away]["goals_conceded"]) / away_matches
        )
    else:
        away_avg_scored.append(0)
        away_avg_conceded.append(0)
        away_win_rate.append(0)
        away_goal_diff.append(0)

    # Update stats after computing stats
    team_stats[home]["matches"] += 1
    team_stats[home]["goals_scored"] += row["home_score"]
    team_stats[home]["goals_conceded"] += row["away_score"]

    team_stats[away]["matches"] += 1
    team_stats[away]["goals_scored"] += row["away_score"]
    team_stats[away]["goals_conceded"] += row["home_score"]

    if row["home_score"] > row["away_score"]:
        team_stats[home]["wins"] += 1
    elif row["home_score"] < row["away_score"]:
        team_stats[away]["wins"] += 1


# Add our features to the dataset
df["home_avg_scored"] = home_avg_scored
df["home_avg_conceded"] = home_avg_conceded
df["away_avg_scored"] = away_avg_scored
df["away_avg_conceded"] = away_avg_conceded
df["home_win_rate"] = home_win_rate
df["away_win_rate"] = away_win_rate
df["home_goal_diff_avg"] = home_goal_diff
df["away_goal_diff_avg"] = away_goal_diff
df["home_advantage"] = (df["neutral"] == False).astype(int)


# Create target variable
def match_result(row):
    if row["home_score"] > row["away_score"]:
        return 2  # Home Win
    elif row["home_score"] < row["away_score"]:
        return 0  # Away Win
    else:
        return 1  # Draw

df["result"] = df.apply(match_result, axis=1)

# Save a new version with averages
df.to_csv("model_training_dataset.csv", index=False)

df["result"] = df.apply(match_result, axis=1)

print("Dataset Info:")
df.info()
print(df.sample(20))
