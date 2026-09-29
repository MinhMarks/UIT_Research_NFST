import numpy as np
import pandas as pd
import torch

from fed_lunar.models.negative_gen import SubspaceNegativeGenerator
from fed_lunar.models.lunar_mlp import KNNDistanceExtractor

train_p = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Train_StandardScaler_data_BoTIoT.csv"
df_tr = pd.read_csv(train_p).select_dtypes(include=[np.number])
X_tr = df_tr[df_tr["label"] == 0].drop(columns=["label"]).values.astype(np.float32)[:2000]

gen = SubspaceNegativeGenerator(negative_ratio=1.0, sigma_pert=0.1, mode="subspace", seed=42)
X_anom, stats = gen.generate(X_tr)

extractor = KNNDistanceExtractor(X_tr, device="cpu")
d_norm = extractor(X_tr[:200], k=10, is_reference_member=True).numpy()
d_synth = extractor(X_anom[:200], k=10, is_reference_member=False).numpy()

print(f"Normal train k-NN mean distances: {np.mean(d_norm, axis=0)}")
print(f"Synthesized negative k-NN mean distances (sigma=0.1): {np.mean(d_synth, axis=0)}")
