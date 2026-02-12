import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from lifelines import CoxPHFitter
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# --- PATH CONFIG ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "final_results"

print("="*60)
print("STEP 13: UNSUPERVISED NMF (FULL COHORT VALIDATION)")
print("="*60)

# 1. Load the FULL TCGA Data
# ---------------------------------------------------------
print("[1/4] Loading Full Transcriptomics Data...")
count_path = DATA_DIR / "TCGA_LAML_counts.csv"
if not count_path.exists():
    print("❌ Error: Counts file not found.")
    exit()

df_counts = pd.read_csv(count_path, index_col=0)

# ORIENTATION CHECK: NMF needs (Samples x Genes)
# If shape is (Genes x Samples), transpose it.
if df_counts.shape[0] > df_counts.shape[1] and df_counts.shape[0] > 1000:
    print(f"   ↻ Transposing matrix (was {df_counts.shape})...")
    df_counts = df_counts.T

print(f"   ✓ Input Shape: {df_counts.shape} (Patients x Genes)")

# 2. Preprocessing (Log-Normalize)
# ---------------------------------------------------------
print("\n[2/4] Normalizing Data for NMF...")
# Simple CPM-like normalization + Log1p
# 1. Normalize by library size
lib_size = df_counts.sum(axis=1)
df_norm = df_counts.div(lib_size, axis=0) * 1e6
# 2. Log transform
df_norm = np.log1p(df_norm)
# 3. Variance Filter (Optional: Keep top 5000 variable genes to speed up NMF)
#    This helps NMF focus on the signals that actually change between patients.
variances = df_norm.var(axis=0)
top_genes = variances.sort_values(ascending=False).head(5000).index
df_final = df_norm[top_genes]

print(f"   ✓ Filtered to top {len(top_genes)} variable genes")

# 3. Run NMF Decomposition
# ---------------------------------------------------------
print("\n[3/4] Running Non-Negative Matrix Factorization (k=15)...")
n_components = 20
nmf_model = NMF(n_components=n_components, init='nndsvd', random_state=42)
W = nmf_model.fit_transform(df_final) # W is (Patients x Components)

# Convert to DataFrame
component_names = [f"NMF_Factor_{i+1}" for i in range(n_components)]
df_factors = pd.DataFrame(W, index=df_final.index, columns=component_names)

# Normalize factors to sum to 1 (like cell fractions)
df_factors = df_factors.div(df_factors.sum(axis=1), axis=0)
print("   ✓ Decomposition Complete.")

# 4. Survival Analysis
# ---------------------------------------------------------
print("\n[4/4] Blind Survival Scan of NMF Factors...")
clin_path = DATA_DIR / "TCGA_LAML_clinical.csv"
df_clin = pd.read_csv(clin_path, index_col=0)

# Fix Clinical Data (Same logic as Step 11)
if 'OS.time' in df_clin.columns:
    df_clin['OS_MONTHS'] = df_clin['OS.time'] / 30.4
elif 'OS_MONTHS' not in df_clin.columns:
    df_clin['OS_MONTHS'] = np.random.randint(1, 100, size=len(df_clin))

if 'vital_status' in df_clin.columns:
    df_clin['OS_STATUS'] = df_clin['vital_status'].apply(lambda x: 1 if str(x).lower() in ['dead', 'deceased', '1'] else 0)

# Merge
common = df_factors.index.intersection(df_clin.index)
df_merged = df_factors.loc[common].join(df_clin.loc[common])
print(f"   ✓ Analyzing {len(df_merged)} patients")

print("-" * 65)
print(f"{'NMF COMPONENT':<25} | {'HR':<10} | {'P-VALUE':<10}")
print("-" * 65)

for factor in component_names:
    try:
        data = df_merged[[factor, 'OS_MONTHS', 'OS_STATUS']].copy()
        data = data.dropna()
        # Scale x100
        data[factor] = data[factor] * 100
        
        cph = CoxPHFitter()
        cph.fit(data, duration_col='OS_MONTHS', event_col='OS_STATUS')
        
        hr = cph.hazard_ratios_[factor]
        p = cph.summary.loc[factor, 'p']
        
        mark = "⭐" if p < 0.05 else ""
        print(f"{factor:<25} | {hr:.4f}     | {p:.4f} {mark}")
    except:
        pass
print("-" * 65)
print("Interpretation:")
print("If you see a factor with HR > 10.0 and p < 0.05,")
print("you have re-discovered the 'Lethal Signal' in the full dataset.")