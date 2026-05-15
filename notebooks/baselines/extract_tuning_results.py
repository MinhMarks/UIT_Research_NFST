import pandas as pd
import os
import glob
import sys

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows Console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def extract_tuning_results():
    # --- Cấu hình lọc dữ liệu ---
    TARGET_NOISE = 0                      # Đổi thành 1, 3, 5... hoặc để None nếu lấy tất cả
    TARGET_SCALER = 'QuantileTransformer' # Đổi scaler tương ứng, hoặc để None nếu lấy tất cả
    TARGET_MODELS = ["LUNAR", "KNN", "LOF", "IForest", "AutoEncoder"]
    # ----------------------------

    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(current_dir, 'outputs')

    # Find all Experiment directories
    experiment_dirs = glob.glob(os.path.join(outputs_dir, 'Experiment_Baseline_Tuning_*'))
    
    if not experiment_dirs:
        print("Không tìm thấy thư mục kết quả (outputs/Experiment_Baseline_Tuning_*).")
        return

    # Lấy thư mục chạy gần nhất dựa trên thời gian
    latest_dir = max(experiment_dirs, key=os.path.getmtime)
    csv_file = os.path.join(latest_dir, 'Tuned_Baseline_Results_All.csv')

    if not os.path.exists(csv_file):
        print(f"Không tìm thấy file kết quả: {csv_file}")
        return

    print(f"Đang đọc file kết quả mới nhất từ: {csv_file}")
    df = pd.read_csv(csv_file)

    # Lọc theo target_models
    df_filtered = df[df['Model'].isin(TARGET_MODELS)].copy()

    # Lọc theo Noise và Scaler nếu có
    if TARGET_NOISE is not None:
        if 'Noise' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Noise'] == TARGET_NOISE]
            print(f"Đã lọc Noise == {TARGET_NOISE}")
            
    if TARGET_SCALER is not None:
        if 'Scaled' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Scaled'] == TARGET_SCALER]
            print(f"Đã lọc Scaled == '{TARGET_SCALER}'")

    if df_filtered.empty:
        print("Không có kết quả nào sau khi áp dụng các bộ lọc (Model, Noise, Scaler).")
        return

    # Sắp xếp Parameters và gắn nhãn "Config 1", "Config 2", "Config 3" cho rõ ràng
    # pd.factorize giúp đánh số các setup duy nhất cho từng model
    df_filtered['Config_ID'] = df_filtered.groupby(['Model'])['Parameters'].transform(lambda x: pd.factorize(x)[0] + 1)
    df_filtered['Config_Name'] = 'Setup ' + df_filtered['Config_ID'].astype(str) + ': ' + df_filtered['Parameters']

    # Tạo bảng Pivot
    # Hàng (Index): Model, Config_Name
    # Cột (Columns): Dataset
    # Giá trị (Values): AUCROC (làm tròn 2 chữ số thập phân)
    pivot_df = pd.pivot_table(
        df_filtered, 
        values='AUCROC', 
        index=['Model', 'Config_Name'], 
        columns='Dataset', 
        aggfunc='mean'
    ).round(2)

    # Lưu ra file CSV
    output_csv = os.path.join(latest_dir, 'Report_Tuning_AUCROC.csv')
    pivot_df.to_csv(output_csv)
    
    print("-" * 80)
    print("Bảng Pivot Result (AUCROC %):")
    print("-" * 80)
    print(pivot_df)
    print("-" * 80)
    print(f"\nĐã xuất kết quả thành công ra file: {output_csv}")
    
    # In ra dạng Latex nếu cần đưa thẳng vào bài
    print("\n[Mã LaTeX dùng cho main.tex nếu bạn cần]")
    try:
        # Tương thích với pandas cũ và mới
        if hasattr(pivot_df.style, "to_latex"):
            print(pivot_df.style.to_latex())
        else:
            print(pivot_df.to_latex())
    except Exception as e:
        print("Không thể in ra dạng Latex, vui lòng copy từ bảng hoặc file CSV.")

if __name__ == "__main__":
    extract_tuning_results()
