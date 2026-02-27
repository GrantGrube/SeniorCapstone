import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Data/Processed/model_training_dataset.csv")

result_counts = df["result"].value_counts().sort_index()

winners = df.groupby('home_team').count()['date']
winners = winners.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(winners.index, winners.values)
plt.xticks(rotation=90)
plt.xlabel('Team')
plt.ylabel('Number of World Cup Games Won')
plt.title('Teams that Have Won the Most World Cup Games')
plt.tight_layout()

plt.figure()
plt.pie(
    result_counts,
    labels=["Away Win", "Draw", "Home Win"],
    autopct="%1.1f%%"
)
plt.title("World Cup Match Result Distribution")
plt.show()
