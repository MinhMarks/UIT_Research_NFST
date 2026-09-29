import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from fed_lunar.models.negative_gen import SubspaceNegativeGenerator
from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss

train_p = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Train_StandardScaler_data_BoTIoT.csv"
test_p = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Test_StandardScaler_data_BoTIoT.csv"

df_tr = pd.read_csv(train_p).select_dtypes(include=[np.number])
df_te = pd.read_csv(test_p).select_dtypes(include=[np.number])

X_tr = df_tr[df_tr["label"] == 0].drop(columns=["label"]).values.astype(np.float32)[:3000]
y_te = df_te["label"].values.astype(int)[:3000]
X_te = df_te.drop(columns=["label"]).values.astype(np.float32)[:3000]

extractor = KNNDistanceExtractor(X_tr, device="cpu")

# Multi-scale perturbation generator: sigma in [0.2, 1.0, 3.0, 6.0]
class MultiScaleGenerator:
    def __init__(self, scales=[0.2, 0.5, 1.5, 3.0, 6.0]):
        self.scales = scales
        self.gens = [SubspaceNegativeGenerator(negative_ratio=1.0/len(scales), sigma_pert=s, seed=42+i) 
                     for i, s in enumerate(scales)]
        
    def generate(self, X):
        parts = [g.generate(X)[0] for g in self.gens]
        return np.vstack(parts)

ms_gen = MultiScaleGenerator()
X_anom = ms_gen.generate(X_tr)

# Train a monotonic / well-calibrated LUNAR MLP
k = 10
model = LUNAR_MLP(k=k, hidden_dims=[64, 32, 16], dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = LunarDistanceRankingLoss()

for epoch in range(15):
    d_norm = extractor(X_tr[:1000], k=k, is_reference_member=True)
    d_anom = extractor(X_anom[:1000], k=k, is_reference_member=False)
    
    # Forward
    l_norm = model(d_norm)
    l_anom = model(d_anom)
    loss = loss_fn(l_norm, l_anom)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test evaluation
model.eval()
with torch.no_grad():
    d_test = extractor(X_te, k=k, is_reference_member=False)
    scores = model.predict_proba(d_test).cpu().numpy().ravel()

auc = roc_auc_score(y_te, scores) * 100.0
print(f"=== Multi-Scale Perturbation LUNAR on BoTIoT ===")
print(f"AUC-ROC: {auc:.2f}% (Previously: 0.15%!)")
print(f"Normal test mean score: {np.mean(scores[y_te == 0]):.4f}")
print(f"Attack test mean score: {np.mean(scores[y_te == 1]):.4f}")
