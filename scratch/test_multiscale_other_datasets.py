import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from fed_lunar.models.negative_gen import SubspaceNegativeGenerator
from fed_lunar.models.lunar_mlp import LUNAR_MLP, KNNDistanceExtractor, LunarDistanceRankingLoss

datasets = ["CICIoT2023", "N_BaIoT"]

class MultiScaleGenerator:
    def __init__(self, scales=[0.2, 0.5, 1.5, 3.0, 6.0]):
        self.scales = scales
        self.gens = [SubspaceNegativeGenerator(negative_ratio=1.0/len(scales), sigma_pert=s, seed=42+i) 
                     for i, s in enumerate(scales)]
        
    def generate(self, X):
        parts = [g.generate(X)[0] for g in self.gens]
        return np.vstack(parts)

for ds in datasets:
    train_p = f"/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Train_StandardScaler_data_{ds}.csv"
    test_p = f"/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Test_StandardScaler_data_{ds}.csv"

    df_tr = pd.read_csv(train_p).select_dtypes(include=[np.number])
    df_te = pd.read_csv(test_p).select_dtypes(include=[np.number])

    X_tr = df_tr[df_tr["label"] == 0].drop(columns=["label"]).values.astype(np.float32)[:4000]
    y_te = df_te["label"].values.astype(int)[:4000]
    X_te = df_te.drop(columns=["label"]).values.astype(np.float32)[:4000]

    extractor = KNNDistanceExtractor(X_tr, device="cpu")
    ms_gen = MultiScaleGenerator()
    X_anom = ms_gen.generate(X_tr)

    k = 10
    model = LUNAR_MLP(k=k, hidden_dims=[64, 32, 16], dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = LunarDistanceRankingLoss()

    for epoch in range(12):
        d_norm = extractor(X_tr[:1000], k=k, is_reference_member=True)
        d_anom = extractor(X_anom[:1000], k=k, is_reference_member=False)
        
        l_norm = model(d_norm)
        l_anom = model(d_anom)
        loss = loss_fn(l_norm, l_anom)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        d_test = extractor(X_te, k=k, is_reference_member=False)
        scores = model.predict_proba(d_test).cpu().numpy().ravel()

    auc = roc_auc_score(y_te, scores) * 100.0
    print(f"=== {ds}: Multi-Scale Perturbation LUNAR ===")
    print(f"AUC-ROC: {auc:.2f}% | Normal score: {np.mean(scores[y_te == 0]):.4f} | Attack score: {np.mean(scores[y_te == 1]):.4f}")
