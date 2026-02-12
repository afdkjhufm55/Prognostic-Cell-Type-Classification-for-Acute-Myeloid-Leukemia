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
print("STEP 4: MERGED DECONVOLUTION (AUTO-ROTATION FIX)")
print("="*60)

# 1. Load Reference & FIX ROTATION
# ---------------------------------------------------------
print("[1/5] Loading Merged Reference...")
ref_path = DATA_DIR / "dtd_reference_merged.joblib"
ref_data = joblib.load(ref_path)
ref_expr = ref_data['ref_expr']

# AUTO-ROTATE: Genes should be Rows (thousands), Cells should be Cols (dozens)
if ref_expr.shape[0] < ref_expr.shape[1]:
    print(f"   ↻ Rotating Reference Matrix (was {ref_expr.shape})...")
    ref_expr = ref_expr.T

print(f"   ✓ Reference: {ref_expr.shape[0]} genes x {ref_expr.shape[1]} cell types")
# Ensure the cell types list matches the columns
ref_cell_types = list(ref_expr.columns)

# 2. Load TCGA Data & FIX ORIENTATION
# ---------------------------------------------------------
print("\n[2/5] Loading TCGA Data...")
data_path = DATA_DIR / "TCGA_LAML_counts.csv"
tcga_df = pd.read_csv(data_path, index_col=0)

# AUTO-ROTATE TCGA: If index looks like ENSG, we want Genes as Columns for now?
# Actually, let's standardize: Genes as COLUMNS for the TCGA DF initially makes pandas easier
if str(tcga_df.index[0]).startswith("ENSG"):
    print("   ↻ Transposing TCGA matrix (Genes -> Columns)...")
    tcga_df = tcga_df.T

gene_cols = [c for c in tcga_df.columns if str(c).startswith("ENSG")]
tcga_counts = tcga_df[gene_cols]
print(f"   ✓ TCGA: {tcga_counts.shape[1]} genes")

# 3. TRANSLATOR (Ensembl -> Symbol)
# ---------------------------------------------------------
print("\n[3/5] Aligning Gene IDs...")
clean_ensembl = [str(g).split('.')[0] for g in tcga_counts.columns]
tcga_counts.columns = clean_ensembl

# Check intersection against Reference Index (Genes)
common = set(tcga_counts.columns) & set(ref_expr.index)

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
    
    # Sum duplicates and keep only Symbols
    symbol_cols = [c for c in tcga_counts.columns if c in mapped_genes.values()]
    tcga_counts = tcga_counts[symbol_cols].groupby(level=0, axis=1).sum()

common_genes = list(set(tcga_counts.columns) & set(ref_expr.index))
print(f"   ✓ Final Common Genes: {len(common_genes):,}")

if len(common_genes) == 0:
    print("❌ CRITICAL ERROR: 0 matching genes. Check reference gene names (Index).")
    print(f"   Ref Index Example: {list(ref_expr.index)[:5]}")
    exit()

# 4. Deconvolution
# ---------------------------------------------------------
print("\n[4/5] Running Deconvolution...")
# X_ref needs to be (Genes x Cells) -> Subset to common
X_ref = ref_expr.loc[common_genes, :].values 

# y_tcga needs to be (Genes x Patients) -> Subset to common & Transpose
y_tcga = tcga_counts[common_genes].values.T 

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
frac_df = pd.DataFrame(fractions, index=tcga_counts.index, columns=ref_cell_types)
frac_df.to_csv(RESULTS_DIR / "TCGA_LAML_cell_fractions.csv")
print("✅ Done.")