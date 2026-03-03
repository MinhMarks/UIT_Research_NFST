import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_models_aucpr(input_csv, output_csv="Best_Baselines_AUCPR.csv", plot_file="Compare_AUCPR.png"):
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}")
        return

    print(f"Reading {input_csv} ...\n")
    df = pd.read_csv(input_csv)

    # Convert AUCPR to numeric just in case
    df['AUCPR'] = pd.to_numeric(df['AUCPR'], errors='coerce')

    # Drop rows where AUCPR is NaN
    df = df.dropna(subset=['AUCPR'])

    if df.empty:
        print("No valid AUCPR data found in the file.")
        return

    # Find the best configuration for each Model, Dataset, Scaler, and Noise level based on AUCPR
    idx = df.groupby(['Dataset', 'Scaled', 'Noise', 'Model'])['AUCPR'].idxmax()
    best_models_df = df.loc[idx].sort_values(by=['Dataset', 'AUCPR'], ascending=[True, False])

    print("=== BEST MODELS BY AUCPR ===")
    display_cols = ['Dataset', 'Model', 'AUCPR', 'AUCROC', 'F1 Score', 'Parameters']
    print(best_models_df[display_cols].to_string(index=False))

    # Save to CSV
    best_models_df.to_csv(output_csv, index=False)
    print(f"\nSaved best AUCPR results to {output_csv}")

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=best_models_df, x='Model', y='AUCPR', hue='Dataset')
    plt.title('Best AUCPR per Model and Dataset')
    plt.ylabel('AUCPR (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Saved AUCPR comparison chart to {plot_file}")

if __name__ == "__main__":
    compare_models_aucpr("Tuned_Baseline_Results_All.csv")
