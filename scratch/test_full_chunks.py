import os
import pandas as pd
import numpy as np

dir_path = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/DataProcessing/edgeiiotset-cyber-security-dataset-of-iot-iiot"

# Let's test reading DDoS_ICMP_Flood_attack.csv or DNN-EdgeIIoT-dataset.csv chunk by chunk completely
file_path = os.path.join(dir_path, "Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv")
print(f"Reading {file_path} fully in chunks of 5000...", flush=True)

peek_iter = pd.read_csv(file_path, index_col=None, header=0, chunksize=5000, on_bad_lines='warn')
count = 0
for chunk in peek_iter:
    count += 1
    if count % 10 == 0:
        print(f"Processed {count} chunks. Shape: {chunk.shape}", flush=True)
print("Done!", flush=True)
