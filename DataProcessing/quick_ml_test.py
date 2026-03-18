import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def eval_dataset(dataset_name="ToNIoT", scaler_name="QuantileTransformer"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'Datascaled', 'Official_OC_Data')
    
    train_path = os.path.join(data_dir, f'Train_{scaler_name}_data_{dataset_name}.csv')
    test_path = os.path.join(data_dir, f'Test_{scaler_name}_data_{dataset_name}.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"⏩ Không tìm thấy bộ đôi file Train/Test cho {dataset_name} ({scaler_name}). Bỏ qua...")
        return
        
    print(f"\n=======================================================")
    print(f"🔍 BẮT ĐẦU TEST BỘ DỮ LIỆU: {dataset_name} (Scaler: {scaler_name})")
    print(f"=======================================================")
    
    # 1. Load Data
    t0 = time.time()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    y_train = train_df.pop('label')
    X_train = train_df
    
    y_test = test_df.pop('label')
    X_test = test_df
    
    print(f"✅ Tải dữ liệu thành công ({time.time() - t0:.2f}s). Train: {X_train.shape[0]} mẫu, Test: {X_test.shape[0]} mẫu")
    
    # Kểm tra nhãn (Label Integrity Check)
    if (y_train == 1).any():
        print("❌ CẢNH BÁO MẠNH: Tập Train chứa nhãn Anomaly (1)! Mô hình One-Class sẽ bị hỏng.")
        return
        
    print("⏳ Đang huấn luyện Isolation Forest (Tìm kiếm phổ vô thường)...")
    t1 = time.time()
    # Contamination = tỉ lệ nhiễu cực thấp trong dữ liệu bình thường.
    model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    
    # Huấn luyện chỉ bằng dữ liệu bình thường (One-Class setup)
    model.fit(X_train)
    print(f"✅ Hoàn tất huấn luyện ({time.time() - t1:.2f}s).")
    
    # 3. Chấm điểm và đưa ra dự đoán
    print("⏳ Đang dự đoán trên tập Test...")
    # Lấy điểm mức độ bất thường (Anomaly Score). -score_samples() trả về giá trị dương là bất thường, âm là bình thường
    y_scores = -model.score_samples(X_test)
    
    # Lấy nhãn dự đoán cứng (predict trả về: 1 (normal) và -1 (anomaly))
    y_pred_sklearn = model.predict(X_test)
    # Map lại thành 0 (normal) và 1 (anomaly) cho phù hợp với test set
    y_pred = np.where(y_pred_sklearn == -1, 1, 0)
    
    # 4. Tính toán Metrics
    auc_roc = roc_auc_score(y_test, y_scores)
    auc_pr = average_precision_score(y_test, y_scores)
    
    print(f"\n📈 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (ISOLATION FOREST)")
    print(f"-------------------------------------------------------")
    print(f" ⭐ AUC-ROC Score: {auc_roc:.4f}  (>0.5 là có học được, >0.8 là rất tốt)")
    print(f" ⭐ AUC-PR Score:  {auc_pr:.4f}  (Quan trọng trong dữ liệu lệch chuẩn)")
    print(f"-------------------------------------------------------")
    print("Bảng chi tiết Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Anomaly (1)"]))
    print(f"=======================================================\n")

if __name__ == "__main__":
    print("🚀 BỘ KIỂM THỬ XÁC THỰC MÁY HỌC DÀNH CHO DỮ LIỆU ONE-CLASS (OC) 🚀")
    print("Mục đích: Xác thực xem các file CSV vừa sinh ra có thực sự học được thuật toán Machine Learning hay không.\n")
    
    # Danh sách các tập dữ liệu sẽ quét
    datasets_to_test = ['ToNIoT', 'N_BaIoT', 'BoTIoT', 'CICIoT2023']
    
    # Chúng ta sử dụng chung một Scaler đại diện (QuantileTransformer) để tiết kiệm thời gian test
    scaler_to_test = "QuantileTransformer"
    
    for ds in datasets_to_test:
        eval_dataset(dataset_name=ds, scaler_name=scaler_to_test)
