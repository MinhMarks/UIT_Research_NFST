import os
import pandas as pd

dir_path = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/DataProcessing/edgeiiotset-cyber-security-dataset-of-iot-iiot"

for root, _, files in os.walk(dir_path):
    for file in files:
        if not file.endswith(".csv"):
            continue
        file_full_path = os.path.join(root, file)
        print(f"Testing file: {file_full_path}", flush=True)
        try:
            peek_iter = pd.read_csv(file_full_path, index_col=None, header=0, chunksize=5000, on_bad_lines='warn')
            chunk = next(peek_iter)
            print(f"  Successfully read first chunk. Shape: {chunk.shape}", flush=True)
        except Exception as e:
            print(f"  Error reading {file_full_path}: {e}", flush=True)
