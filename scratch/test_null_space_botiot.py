import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

train_p = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Train_StandardScaler_data_BoTIoT.csv"
test_p = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data/Test_StandardScaler_data_BoTIoT.csv"

df_tr = pd.read_csv(train_p).select_dtypes(include=[np.number])
df_te = pd.read_csv(test_p).select_dtypes(include=[np.number])

X_tr = df_tr[df_tr["label"] == 0].drop(columns=["label"]).values.astype(np.float32)[:5000]
y_te = df_te["label"].values.astype(int)[:5000]
X_te = df_te.drop(columns=["label"]).values.astype(np.float32)[:5000]

mu = np.mean(X_tr, axis=0)
X_c = X_tr - mu
u, s, vh = np.linalg.svd(X_c, full_matrices=True)
r = int(np.sum(s > 1e-4))
W_null = vh[r:].T

proj_te = np.sum(((X_te - mu) @ W_null) ** 2, axis=1)
auc_null = roc_auc_score(y_te, proj_te) * 100.0
print(f"BoTIoT Null Space AUC: {auc_null:.2f}% (rank r={r}/{X_tr.shape[1]})")
