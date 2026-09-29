import numpy as np
import pandas as pd
import torch

train_p = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Train_StandardScaler_data_BoTIoT.csv"
test_p = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Test_StandardScaler_data_BoTIoT.csv"

df_tr = pd.read_csv(train_p).select_dtypes(include=[np.number])
df_te = pd.read_csv(test_p).select_dtypes(include=[np.number])

X_tr = df_tr[df_tr["label"] == 0].drop(columns=["label"]).values.astype(np.float32)[:2000]
y_te = df_te["label"].values.astype(int)[:2000]
X_te = df_te.drop(columns=["label"]).values.astype(np.float32)[:2000]

# Compute k-NN distances to X_tr for normal test points vs attack test points
from fed_lunar.models.lunar_mlp import KNNDistanceExtractor
extractor = KNNDistanceExtractor(X_tr, device="cpu")

d_norm = extractor(X_te[y_te == 0], k=10).numpy()
d_anom = extractor(X_te[y_te == 1], k=10).numpy()

print(f"Normal test count: {len(d_norm)}, Attack test count: {len(d_anom)}")
print(f"Normal test mean k-NN dists: {np.mean(d_norm, axis=0)}")
print(f"Attack test mean k-NN dists: {np.mean(d_anom, axis=0)}")
print(f"Normal test min/max: min={np.min(d_norm):.3f}, max={np.max(d_norm):.3f}")
print(f"Attack test min/max: min={np.min(d_anom):.3f}, max={np.max(d_anom):.3f}")
