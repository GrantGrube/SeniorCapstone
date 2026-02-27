import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Load dataset
results = pd.read_csv("Data/International Football Results from 1872 - 2026/results.csv")
goalscorers = pd.read_csv("Data/International Football Results from 1872 - 2026/goalscorers.csv")
shootouts = pd.read_csv("Data/International Football Results from 1872 - 2026/shootouts.csv")
# Perform basic cleaning on the data

# Ensure that there is a proper date format
results["date"] = pd.to_datetime(results["date"])
goalscorers["date"] = pd.to_datetime(goalscorers["date"])
shootouts["date"] = pd.to_datetime(shootouts["date"])

# Ensure we have numeric scores
results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")

# Drop rows with missing values
results = results.dropna(subset=["home_score", "away_score"])

# Filter only by World Cup Matches

results = results[results["tournament"] == "FIFA World Cup"]
print("Total World Cup matches:", len(results))


# Creating a target variable
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2  # Home Win
    elif row["home_score"] < row["away_score"]:
        return 0  # Away Win
    else:
        return 1  # Draw


results["result"] = results.apply(get_result, axis=1)

# Select specific columns that we need

formatted_df = results[[
    "date",
    "home_team",
    "away_team",
    "neutral",
    "home_score",
    "away_score",
    "result"
]].copy()

# Sort by Chronological order
formatted_df = formatted_df.sort_values("date").reset_index(drop=True)

# Filter Goalscorers & Shootouts to World Cup Matches
wc_matches = formatted_df[["date", "home_team", "away_team"]]

goalscorers = goalscorers.merge(
    wc_matches,
    on=["date", "home_team", "away_team"],
    how="inner"
)

shootouts = shootouts.merge(
    wc_matches,
    on=["date", "home_team", "away_team"],
    how="inner"
)

# Aggregate goalscorers
goal_summary = goalscorers.groupby(
    ["date", "home_team", "away_team"]
).size().reset_index(name="total_goals_recorded")

formatted_df = formatted_df.merge(
    goal_summary,
    on=["date", "home_team", "away_team"],
    how="left"
)

formatted_df["total_goals_recorded"] = formatted_df["total_goals_recorded"].fillna(0)

# Merge Penalty Shootouts
formatted_df = formatted_df.merge(
    shootouts,
    on=["date", "home_team", "away_team"],
    how="left"
)

# Rename winner column if present
if "winner" in formatted_df.columns:
    formatted_df.rename(columns={"winner": "shootout_winner"}, inplace=True)

# Create shootout indicator
formatted_df["went_to_shootout"] = formatted_df["shootout_winner"].notna().astype(int)

# Validate Goal integrity

formatted_df["score_sum"] = (
    formatted_df["home_score"] + formatted_df["away_score"]
)

formatted_df["goal_mismatch"] = (
    formatted_df["score_sum"] != formatted_df["total_goals_recorded"]
)

print("Goal mismatches:", formatted_df["goal_mismatch"].sum())


# Save clean version
# output_path = BASE_DIR / "world_cup_only_dataset.csv"
# formatted_df.to_csv(output_path, index=False)

print("\nWorld Cup dataset created successfully.")
print("Final dataset shape:", formatted_df.shape)

print("\nDataset Info:")
formatted_df.info()

print("\nSample rows:")
print(formatted_df.sample(20))
