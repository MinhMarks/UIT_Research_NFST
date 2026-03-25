import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')

_script_dir = os.path.dirname(os.path.abspath(__file__))
OC_DATA_DIR = os.path.normpath(os.path.join(_script_dir, '..', 'Datascaled', 'Official_OC_Data'))
ANOMALY_DATA_DIR = os.path.normpath(os.path.join(_script_dir, '..', 'Datascaled', 'Official_Anomaly_Data'))
PICS_DIR = os.path.normpath(os.path.join(_script_dir, '..', 'pictures'))
os.makedirs(PICS_DIR, exist_ok=True)

def load_and_pca(train_path, test_path, n_samples=2000):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Use Test set for visualization because it contains both Normal (0) and Anomaly (1)
    # Downsample for faster plotting
    df = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)
    
    y = df.pop('label').values
    X = df.to_numpy()
    
    # Guaranteed no nans
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    return pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Label': y})

def plot_distributions(dataset="BoTIoT", scaler="MinMaxScaler"):
    print(f"Testing Distributions for {dataset} ({scaler})...")
    
    # Original Data
    real_train = os.path.join(OC_DATA_DIR, f"Train_{scaler}_data_{dataset}.csv")
    real_test = os.path.join(OC_DATA_DIR, f"Test_{scaler}_data_{dataset}.csv")
    df_real = load_and_pca(real_train, real_test)
    
    if df_real is None:
        print(f"Original files not found for {dataset}")
        return
        
    df_real['Type'] = 'Original'
    
    # Local Data
    loc_train = os.path.join(ANOMALY_DATA_DIR, f"Train_local_{scaler}_data_{dataset}.csv")
    loc_test = os.path.join(ANOMALY_DATA_DIR, f"Test_local_{scaler}_data_{dataset}.csv")
    df_local = load_and_pca(loc_train, loc_test)
    if df_local is not None: df_local['Type'] = 'Local (Synthetic)'
    
    # Cluster Data
    clnt_train = os.path.join(ANOMALY_DATA_DIR, f"Train_cluster_{scaler}_data_{dataset}.csv")
    clnt_test = os.path.join(ANOMALY_DATA_DIR, f"Test_cluster_{scaler}_data_{dataset}.csv")
    df_cluster = load_and_pca(clnt_train, clnt_test)
    if df_cluster is not None: df_cluster['Type'] = 'Cluster (Synthetic)'
        
    # Global Data
    glob_train = os.path.join(ANOMALY_DATA_DIR, f"Train_global_{scaler}_data_{dataset}.csv")
    glob_test = os.path.join(ANOMALY_DATA_DIR, f"Test_global_{scaler}_data_{dataset}.csv")
    df_global = load_and_pca(glob_train, glob_test)
    if df_global is not None: df_global['Type'] = 'Global (Synthetic)'

    # Combine all available DataFrames
    dfs = [df for df in [df_real, df_local, df_cluster, df_global] if df is not None]
    if len(dfs) < 2:
        print("Not enough synthetic files found to compare.")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, df_plot in enumerate(dfs):
        sns.scatterplot(
            data=df_plot, x='PCA1', y='PCA2', hue='Label', 
            palette={0: "blue", 1: "red"}, alpha=0.6, ax=axes[i], marker="o", s=15
        )
        axes[i].set_title(df_plot['Type'].iloc[0], fontsize=14, fontweight='bold')
        title_map = {0: 'Normal', 1: 'Anomaly'}
        
        # Clean up legend
        handles, labels = axes[i].get_legend_handles_labels()
        if labels:
            axes[i].legend([handles[0], handles[1]], ['Normal', 'Anomaly'], title='Class')
            
    plt.tight_layout()
    
    out_path = os.path.join(PICS_DIR, f"Anomaly_Distributions_{dataset}_{scaler}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Tester successfully generated 2D PCA plots at: {out_path}")

if __name__ == "__main__":
    import sys
    import numpy as np # Ensure loaded by PCA
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "BoTIoT"
    sc_name = sys.argv[2] if len(sys.argv) > 2 else "MinMaxScaler"
    
    plot_distributions(ds_name, sc_name)
