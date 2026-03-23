import os
import glob
import pandas as pd

def merge_csvs():
    # Thư mục chứa các file csv hiện tại
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'outputs')
    
    print(f"Đang tìm kiếm tất cả các file CSV kết quả trong {output_dir}...")
    
    # Tìm tất cả các file csv trong thư mục outputs (bao gồm cả thư mục con)
    all_csvs = glob.glob(os.path.join(output_dir, '**', '*.csv'), recursive=True)
    
    # Lọc ra các file của fast_baselines (có column Model, Dataset, Scaler, AUCROC)
    valid_csvs = []
    for f in all_csvs:
        # Bỏ qua file tổng hợp đã tạo trước đó để tránh lặp
        if 'all_fast_baseline' in os.path.basename(f).lower():
            continue
        try:
            df = pd.read_csv(f)
            cols = [c.lower() for c in df.columns]
            if 'model' in cols and 'aucroc' in cols and 'dataset' in cols:
                valid_csvs.append((f, df))
        except Exception:
            pass
            
    if not valid_csvs:
        print("Không tìm thấy file kết quả rời rạc nào để ghép!")
        return
        
    print(f"Bắt đầu ghép {len(valid_csvs)} file...")
    
    df_list = [df for _, df in valid_csvs]
    final_df = pd.concat(df_list, ignore_index=True)
    
    # Lưu file ghép
    out_path = os.path.join(output_dir, 'all_fast_baseline_results_merged.csv')
    final_df.to_csv(out_path, index=False)
    
    print(f"Thành công! Đã gộp {len(final_df)} dòng vào file duy nhất:")
    print(f"-> {out_path}")
    print("Bạn có thể dùng file này cho generate_best_results_report.py")

if __name__ == '__main__':
    merge_csvs()
