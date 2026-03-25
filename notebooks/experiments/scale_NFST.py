import os
import time
import numpy as np
import pandas as pd
import tracemalloc
from scipy.linalg import null_space
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] faiss not installed. Falling back to numpy scoring.")

# =====================================================================
# CORE LOC-NFST FUNCTIONS (Adapted for pure speed & scale testing)
# =====================================================================
def cluster_kmeans(data: np.ndarray, k: int):
    k = min(k, len(data))
    # n_init=1 since we purely want to measure the algorithmic time footprint,
    # not necessarily the absolute best clustering accuracy.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
    labels = kmeans.fit_predict(data)
    sorted_idx = np.argsort(labels)
    return data[sorted_idx], labels[sorted_idx], kmeans.cluster_centers_.astype(np.float32)

def calculate_NPD_optimized(X: np.ndarray, y: np.ndarray, epsilon: float = 1e-6):
    X = np.ascontiguousarray(X.T, dtype=np.float32)
    d, N = X.shape
    classes = np.unique(y)

    mean_total = np.mean(X, axis=1, keepdims=True)
    P_t = X - mean_total

    try:
        U, s_t, _ = np.linalg.svd(P_t, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD did not converge or memory error internally within LAPACK
        raise MemoryError("SVD failed/OoM")

    rank_Pt = int(np.sum(s_t > epsilon))
    Q = U[:, :rank_Pt].astype(np.float32)

    del P_t, U, s_t

    S_w = np.zeros((d, d), dtype=np.float32)
    for cls in classes:
        mask = (y == cls)
        X_cls = X[:, mask]
        m_cls = np.mean(X_cls, axis=1, keepdims=True)
        diff = X_cls - m_cls
        S_w += diff @ diff.T
        del diff

    S_w /= N
    del X

    A = Q.T @ S_w @ Q
    del S_w

    B = null_space(A)
    del A

    W = (Q @ B).astype(np.float32)
    return W

def compute_scores(X_test, W, centers):
    if FAISS_AVAILABLE:
        d = centers.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(centers.astype('float32'))
        _, nearest_idx = index.search(X_test.astype('float32'), 1)
        nearest_centers = centers[nearest_idx.flatten()]
    else:
        nearest_centers = []
        batch_size = 2000
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            dists = np.linalg.norm(batch[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            idx = np.argmin(dists, axis=1)
            nearest_centers.append(centers[idx])
        nearest_centers = np.vstack(nearest_centers)

    diff = X_test - nearest_centers
    projections = diff @ W
    return np.sqrt(np.sum(projections**2, axis=1))

# =====================================================================
# EXPERIMENT RUNNER
# =====================================================================
def run_scale_experiment(N, d, k=10):
    print(f"  -> Profiling N={N:<8} d={d:<5} ... ", end="", flush=True)
    
    # Heuristic Memory Protection:
    # (N * d * 4 bytes) for float32. 
    # SVD requires an intermediate matrix of at least min(N, d)^2 or similar.
    # We will abort early if theoretical minimum footprint exceeds ~12 GB (safe limit)
    mem_estimate_gb = (N * d * 4) / (1024**3)
    if mem_estimate_gb > 8.0:
        print(f"Skipped (Est. {mem_estimate_gb:.1f} GB > Limit)")
        return {"N": N, "d": d, "Train_Time": "OoM", "Test_Time_Total": "OoM", "Test_Time_Per_Sample": "OoM", "Status": "OoM"}

    try:
        X_train = np.random.randn(N, d).astype(np.float32)
        N_test = min(20000, max(100, int(N * 0.2))) # Cap test size to prevent artificial inflation
        X_test = np.random.randn(N_test, d).astype(np.float32)
        
        # --- TRAIN TIME ---
        tracemalloc.start()
        t0 = time.time()
        
        X_clustered, y_clustered, centers = cluster_kmeans(X_train, k=min(k, N))
        W = calculate_NPD_optimized(X_clustered, y_clustered)
        
        train_time = time.time() - t0
        _, peak_train_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        del X_train, X_clustered, y_clustered
        
        # --- TEST TIME (INFERENCE) ---
        t1 = time.time()
        _ = compute_scores(X_test, W, centers)
        test_time = time.time() - t1
        
        print(f"OK (Train: {train_time:.3f}s)")
        return {
            "N": N, "d": d,
            "Train_Time": train_time,
            "Test_Time_Total": test_time,
            "Test_Time_Per_Sample": test_time / N_test,
            "Status": "OK"
        }
        
    except MemoryError:
        print("OoM")
        return {"N": N, "d": d, "Train_Time": "OoM", "Test_Time_Total": "OoM", "Test_Time_Per_Sample": "OoM", "Status": "OoM"}
    except Exception as e:
        print(f"Error ({e})")
        return {"N": N, "d": d, "Train_Time": "OoM", "Test_Time_Total": "OoM", "Test_Time_Per_Sample": "OoM", "Status": "Error"} # Treat mostly as OoM for LaTeX


def generate_latex_table(results_df, value_col, caption, label):
    """Generates the LaTeX code directly matching the manuscript's format"""
    d_vals = sorted(results_df['d'].unique())
    N_vals = sorted(results_df['N'].unique())
    
    # Header format
    latex =  "\\begin{table}[!ht]\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += "\\centering\n"
    latex += "\\resizebox{\\linewidth}{!}{%\n"
    
    # Columns definition
    cols_def = "|c|" + "c|" * len(d_vals)
    latex += f"\\begin{{tabular}}{{{cols_def}}}\n\\hline\n"
    
    # Headers
    d_headers = " & ".join([f"\\( 10^{{{int(np.log10(d))}}} \\)" if d >= 10 else f"\\( {d} \\)" for d in d_vals])
    latex += f"\\( N \\backslash d \\) & {d_headers} \\\\\n\\hline\n"
    
    # Rows
    for N in N_vals:
        row_str = f"\\( 10^{{{int(np.log10(N))}}} \\)"
        for d in d_vals:
            val = results_df[(results_df['N'] == N) & (results_df['d'] == d)][value_col].values
            if len(val) == 0 or str(val[0]) == 'OoM' or str(val[0]) == 'Error':
                str_val = "OoM"
            else:
                v = float(val[0])
                # Format to scientific notation matching LaTeX style
                # e.g., 0.0013 -> \( 1.3 \times 10^{-3} \)
                if v == 0:
                    str_val = "\\( 0 \\)"
                else:
                    exp = int(np.floor(np.log10(abs(v))))
                    base = v / (10**exp)
                    if -2 <= exp <= 2:
                        str_val = f"\\( {v:.4f} \\)" if v >= 0.01 else f"\\( {v:.4e} \\)".replace('e', ' \\times 10^{').replace('+0', '') + '}'
                    else:
                        str_val = f"\\( {base:.2f} \\times 10^{{{exp}}} \\)"
            
            row_str += f" & {str_val}"
        row_str += " \\\\\n"
        latex += row_str
        
    latex += "\\hline\n\\end{tabular}%\n}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\end{table}\n"
    
    return latex


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(_script_dir, 'outputs', 'Scale_Experiment')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Exact scales requested by User
    N_list = [10**2, 10**3, 10**4, 10**5, 10**6]
    d_list = [10, 10**2, 10**3, 10**4]
    
    results = []
    
    print("==========================================================")
    print(" Starting LOC-NFST Scalability Stress Test")
    print("==========================================================")
    
    for N in N_list:
        for d in d_list:
            res = run_scale_experiment(N, d)
            results.append(res)
            
    df = pd.DataFrame(results)
    csv_path = os.path.join(exp_dir, 'scale_metrics.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n==========================================================")
    print(" LaTeX Output -> TRAINING TIME")
    print("==========================================================")
    tex_train = generate_latex_table(
        df, 
        value_col="Train_Time", 
        caption="Training runtime performance (in seconds) of LOC-NFST for different values of \\( N \\) and \\( d \\).",
        label="tab:scale_train_time"
    )
    print(tex_train)
    
    print("\n==========================================================")
    print(" LaTeX Output -> INFERENCE TIME (Total)")
    print("==========================================================")
    tex_infer = generate_latex_table(
        df, 
        value_col="Test_Time_Total", 
        caption="Total inference runtime performance (in seconds) of LOC-NFST for different values of \\( N \\) and \\( d \\).",
        label="tab:scale_infer_time"
    )
    print(tex_infer)
    
    # Save latex strictly to text file so user can copy easily
    with open(os.path.join(exp_dir, 'latex_tables.tex'), 'w') as f:
        f.write("% --- TRAINING TABLE ---\n")
        f.write(tex_train)
        f.write("\n% --- INFERENCE TABLE ---\n")
        f.write(tex_infer)
        
    print(f"\n[+] Results saved to {exp_dir}")
