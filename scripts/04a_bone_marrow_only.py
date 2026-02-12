import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# --- PATH CONFIG ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "final_results"

print("="*60)
print("STEP 4A: BONE MARROW ONLY (WITH GENE TRANSLATION)")
print("="*60)

# 1. Load Bone Marrow Reference
# ---------------------------------------------------------
print("[1/5] Loading Bone Marrow Reference...")
ref_path = DATA_DIR / "dtd_reference_bm_only.joblib"
if not ref_path.exists():
    print(f"❌ Reference not found: {ref_path}")
    exit()

ref_data = joblib.load(ref_path)
ref_expr = ref_data['ref_expr']
# Use single quotes inside f-string
print(f"   ✓ Reference: {ref_expr.shape[1]} genes (Type: {ref_expr.columns[0]})")

# 2. Load TCGA Data & FIX ORIENTATION
# ---------------------------------------------------------
print("\n[2/5] Loading TCGA Data...")
data_path = DATA_DIR / "TCGA_LAML_counts.csv"
tcga_df = pd.read_csv(data_path, index_col=0)

# CHECK ORIENTATION: If index starts with ENSG, we must transpose
if tcga_df.index[0].startswith("ENSG"):
    print("   ↻ Transposing matrix (Genes -> Columns)...")
    tcga_df = tcga_df.T

# Ensure columns are clean
gene_cols = [c for c in tcga_df.columns if c.startswith("ENSG")]
tcga_counts = tcga_df[gene_cols]
print(f"   ✓ TCGA: {tcga_counts.shape[1]} genes")

# 3. TRANSLATOR (Ensembl -> Symbol)
# ---------------------------------------------------------
print("\n[3/5] Aligning Gene IDs...")
clean_ensembl = [g.split('.')[0] for g in tcga_counts.columns]
tcga_counts.columns = clean_ensembl

# Check intersection
common = set(tcga_counts.columns) & set(ref_expr.columns)
print(f"   Direct match count: {len(common)}")

if len(common) < 500:
    print("   ⚠️ Mismatch detected! Initiating Gene Translation...")
    genes_to_map = list(tcga_counts.columns)
    mapped_genes = {}
    
    batch_size = 1000
    for i in range(0, len(genes_to_map), batch_size):
        print(f"      Translating batch {i}-{min(i+batch_size, len(genes_to_map))}...", end='\r')
        batch = genes_to_map[i:i+batch_size]
        try:
            res = requests.post(
                "https://mygene.info/v3/query",
                json={"q": batch, "scopes": "ensembl.gene", "fields": "symbol", "species": "human"}
            )
            for hit in res.json():
                if "symbol" in hit:
                    mapped_genes[hit["query"]] = hit["symbol"]
        except: pass
            
    print(f"\n   ✓ Translated {len(mapped_genes)} genes")
    tcga_counts = tcga_counts.rename(columns=mapped_genes)
    # Sum duplicates
    symbol_cols = [c for c in tcga_counts.columns if c in mapped_genes.values()]
    tcga_counts = tcga_counts[symbol_cols].groupby(level=0, axis=1).sum()

common_genes = list(set(tcga_counts.columns) & set(ref_expr.columns))
print(f"   ✓ Final Common Genes: {len(common_genes):,}")

# 4. Deconvolution
# ---------------------------------------------------------
print("\n[4/5] Running Deconvolution (BM Only)...")
X_ref = ref_expr.loc[:, common_genes].T.values 
y_tcga = tcga_counts[common_genes].T.values

n_samples = y_tcga.shape[1]
n_celltypes = X_ref.shape[1]
fractions = np.zeros((n_samples, n_celltypes))
model = LinearRegression(positive=True, fit_intercept=False)

# Normalize
X_ref_norm = X_ref / (X_ref.sum(axis=0, keepdims=True) + 1e-6) * 1e6
y_tcga_norm = y_tcga / (y_tcga.sum(axis=0, keepdims=True) + 1e-6) * 1e6

for i in range(n_samples):
    if i % 20 == 0: print(f"   Processing patient {i}/{n_samples}...", end="\r")
    if np.sum(y_tcga_norm[:, i]) > 0:
        model.fit(X_ref_norm, y_tcga_norm[:, i])
        coef = model.coef_
        if np.sum(coef) > 0:
            fractions[i, :] = coef / np.sum(coef)

# 5. Save
# ---------------------------------------------------------
print("\n[5/5] Saving Results...")
RESULTS_DIR.mkdir(exist_ok=True)
frac_df = pd.DataFrame(fractions, index=tcga_counts.index, columns=ref_data['cell_types'])
frac_df.to_csv(RESULTS_DIR / "TCGA_LAML_BM_only_fractions.csv")
print("✅ Done.")