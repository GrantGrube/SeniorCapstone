import requests
import csv
from io import StringIO

url = "https://www.eloratings.net/World.tsv?_=1774309604123"
response = requests.get(url)


tsv_data = StringIO(response.text)
reader = csv.reader(tsv_data, delimiter="\t")


with open("ratings.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in reader:
        rating = row[1]
        team = row[2]
        ELO = row[3]
        Total = row[25]
        print(rating, team, ELO, Total)
        writer.writerow([rating, team, ELO])

#.split \t function to split data into rows to scrape


    
print("Saved to ratings.csv")