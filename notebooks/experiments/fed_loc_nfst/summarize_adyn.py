"""
Summarize ADYN-LOC-NFST vs Static-K Benchmark Results
"""
import glob
import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_csvs = glob.glob(os.path.join(base_dir, 'outputs', 'adyn_results', '*.csv'))
    if not all_csvs:
        print("No CSV files found!")
        return

    # Find the CSV with the largest number of rows (the full benchmark run)
    scored = []
    for f in all_csvs:
        try:
            nrows = len(pd.read_csv(f))
            scored.append((nrows, f))
        except Exception:
            pass

    scored.sort(key=lambda x: x[0], reverse=True)
    max_rows, latest = scored[0]
    print(f"=== Selected largest benchmark: {os.path.basename(latest)} ({max_rows} rows) ===")
    df = pd.read_csv(latest)
    print(f"Total rows: {len(df)}")

    summary = []
    for (ds, sc), grp in df.groupby(['dataset', 'scaler']):
        st = grp[grp['method'].str.startswith('Static')]
        ad = grp[grp['method'].str.startswith('ADYN')]
        if not st.empty and not ad.empty:
            best_st = st.loc[st['AUCROC'].idxmax()]
            ad_row = ad.iloc[0]
            delta = ad_row['AUCROC'] - best_st['AUCROC']
            summary.append({
                'Dataset': ds.replace('data_', ''),
                'Scaler': sc,
                'Static_best_K': int(best_st['K_final']),
                'Static_AUC': round(best_st['AUCROC'], 2),
                'ADYN_K': int(ad_row['K_final']),
                'ADYN_AUC': round(ad_row['AUCROC'], 2),
                'Delta_AUC': round(delta, 2),
                'Static_L': int(best_st['L']),
                'ADYN_L': int(ad_row['L']),
            })

    if summary:
        sdf = pd.DataFrame(summary)
        print("\n" + "="*85)
        print("ADYN-LOC-NFST vs Static-K Comprehensive Comparison Table:")
        print("="*85)
        print(sdf.to_string(index=False))

        print("\n" + "="*85)
        print("Aggregate Statistics:")
        print(f"Average Static AUC : {sdf['Static_AUC'].mean():.2f}%")
        print(f"Average ADYN AUC   : {sdf['ADYN_AUC'].mean():.2f}%")
        wins = np_wins = len(sdf[sdf['Delta_AUC'] > 0])
        ties = len(sdf[sdf['Delta_AUC'] == 0])
        losses = len(sdf[sdf['Delta_AUC'] < 0])
        print(f"ADYN Outperforms Static : {wins} cases")
        print(f"Static Outperforms ADYN : {losses} cases")
        print("="*85)
    else:
        print("Could not group static and adyn pairs.")

if __name__ == '__main__':
    main()
