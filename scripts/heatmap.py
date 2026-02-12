import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# --- PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("="*60)
print("GENERATING WIDESCREEN HEATMAP")
print("="*60)

# 1. RE-CALCULATE NMF (Quickly)
# ---------------------------------------------------------
print("[1/3] Preparing Data...")
count_path = DATA_DIR / "TCGA_LAML_counts.csv"
df_counts = pd.read_csv(count_path, index_col=0)
if df_counts.shape[0] > df_counts.shape[1]: df_counts = df_counts.T

# Normalize
lib_size = df_counts.sum(axis=1)
df_norm = np.log1p(df_counts.div(lib_size, axis=0) * 1e6)
df_final = df_norm[df_norm.var(axis=0).sort_values(ascending=False).head(5000).index]

# Factorize
n_comp = 15
nmf = NMF(n_components=n_comp, init='nndsvd', random_state=42)
W = nmf.fit_transform(df_final)
df_factors = pd.DataFrame(W, index=df_final.index, columns=[f"NMF_Factor_{i+1}" for i in range(n_comp)])
# Normalize rows to sum to 1
df_factors = df_factors.div(df_factors.sum(axis=1), axis=0)

# 2. PLOT WIDESCREEN HEATMAP
# ---------------------------------------------------------
print("[2/3] Plotting...")

# Sort by Factor 1 (The Lethal Factor) to show the pattern clearly
sort_col = "NMF_Factor_1"
df_sorted = df_factors.sort_values(sort_col, ascending=False)

# SET UP THE CANVAS: 24 inches wide x 10 inches tall
plt.figure(figsize=(24, 10)) 

# Draw Heatmap
# - xticklabels=False: Hides the messy patient IDs
# - yticklabels=True: Shows the Factor names
# - cbar_kws: Adjusts the color bar so it doesn't look weird
ax = sns.heatmap(df_sorted.T, 
                 cmap="viridis", 
                 xticklabels=False, 
                 yticklabels=True,
                 cbar_kws={"fraction": 0.01, "pad": 0.01})

# Polish the labels
plt.yticks(rotation=0, fontsize=14, fontweight='bold') # Horizontal labels
plt.ylabel("Inferred Factors", fontsize=16)
plt.xlabel(f"Patients (n={len(df_sorted)}) - Sorted by Risk", fontsize=16)
plt.title("Unsupervised NMF Patient Stratification (High-Res)", fontsize=20, pad=20)

# 3. SAVE
# ---------------------------------------------------------
output_path = FIG_DIR / "Fig2_NMF_Heatmap_Wide.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300) # 300 DPI for high quality
print(f"   ✓ Saved: {output_path.name}")
print("✅ Done.")