import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/Processed/model_training_dataset.csv")

# Create win columns
df["home_win"] = df["home_score"] > df["away_score"]
df["away_win"] = df["away_score"] > df["home_score"]

# Count wins
home_wins = df[df["home_win"]].groupby("home_team").size()
away_wins = df[df["away_win"]].groupby("away_team").size()

# Combine
total_wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)

# Plot Top 10
top10 = total_wins.head(10)

plt.figure()
plt.bar(top10.index, top10.values)
plt.xticks(rotation=90)
plt.title("Top 10 Nations by World Cup Games Won")
plt.xlabel("Team")
plt.ylabel("Total Wins")
plt.show()
