import openpyxl
import json
from openpyxl import workbook
from openpyxl import load_workbook

book = load_workbook(filename="Capstone Sheet.xlsx") #loading the sheet we want to use

print(book.sheetnames)
sheetn = input("Please enter the name of the sheet you would like to use: ")
current = book[sheetn]
print("you have chosen " + current.title)


cell_data = []

for row in current.iter_rows():
    for cell in row:
        cell_data.append({
            "row": cell.row,
            "col": cell.column,
            "coordinate": cell.coordinate,
            "value": cell.value,
            "type": type(cell.value).__name__
        })

output_path = r"\Users\Grant\Desktop\Capstone\UnityCapstone\Assets\StreamingAssets/excel_data.json"



with open(output_path, "w") as f:
    json.dump(cell_data, f, indent=4)

