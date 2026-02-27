import numpy as np
from pyod.models.knn import KNN
from sklearn.metrics import roc_auc_score, average_precision_score

print('--- Test PyOD KNN ---')
# Create synthetic data: normal points near origin, anomalies far away
X_train = np.random.randn(100, 2) * 0.1
X_test = np.vstack([np.random.randn(50, 2) * 0.1, np.random.randn(10, 2) * 5 + 5])

# In original user format: Normal=1, Anomaly=0
y_test_original = np.array([1]*50 + [0]*10)

knn = KNN()
# Should we fit on mostly Normal (1) or all data? PyOD assumes mostly normal.
# But wait... PyOD doesn't take labels in .fit()! It takes just X
knn.fit(X_train) 

# Test prediction
y_pred = knn.predict(X_test) # PyOD outputs 0 for normal, 1 for anomaly
y_proba = knn.predict_proba(X_test) # Column 1 = anomaly probability

print('Example y_test_original (User):', y_test_original[-15:])
print('Example y_pred (PyOD):        ', y_pred[-15:])
print('Example y_proba[:,1] (PyOD):  ', np.round(y_proba[:,1][-15:], 2))

print('\nInverted y_true (y_true_inverted = 1 - y_test_original):')
y_true_inverted = 1 - y_test_original
print('AUCROC on y_proba[:, 1]:', roc_auc_score(y_true_inverted, y_proba[:, 1]))
print('AUCPR on y_proba[:, 1]:', average_precision_score(y_true_inverted, y_proba[:, 1]))

print('\nBUT WHAT IF PyOD thinks 0=Normal and 1=Anomaly during semi-supervised fit(X, y)?')
# If a model like DevNet uses labels during fit...
# Wait, PyOD DevNet takes X, y in fit. IF y=0 is passed for Anomaly instead of y=1...
# Let's inspect tune_baselines.py lines 207-208:
# if model_name == 'DevNet':
#     model.fit(X_train, y_train)
# If y_train is passed exactly as User's labels (0=Anomaly, 1=Normal),
# DevNet will treat 1 as Anomaly! This fundamentally flips what it learns!
