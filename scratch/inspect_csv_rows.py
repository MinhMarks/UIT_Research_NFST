import os
import csv

file_path = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/DataProcessing/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"

print("Reading rows from 104,990 to 105,050 using standard csv module...")
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if 104990 <= i <= 105050:
            print(f"Row {i}: Length {len(row)} | Snippet: {row[:5]} ... {row[-3:]}")
        if i > 105050:
            break
print("Done reading!")
