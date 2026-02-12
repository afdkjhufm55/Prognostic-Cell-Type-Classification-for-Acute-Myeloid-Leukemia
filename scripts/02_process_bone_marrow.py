import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
import joblib
from pathlib import Path

print("="*60)
print("STEP 2: Bone Marrow Single-Cell Reference (Memory Efficient)")
print("="*60)

sc_path = Path("data/bone_marrow_sc.h5ad")
if not sc_path.exists():
    print(f"[ERROR] {sc_path} not found!")
    exit(1)

print("  [1/5] Loading bone marrow atlas...")
sc_data = sc.read_h5ad(sc_path)
print(f"  Raw: {sc_data.n_obs:,} cells x {sc_data.n_vars:,} genes")

if 'cell_type' not in sc_data.obs.columns:
    print("[ERROR] No 'cell_type' column!")
    print("Available columns:", list(sc_data.obs.columns))
    exit(1)

print("  [2/5] Preprocessing...")
sc.pp.filter_genes(sc_data, min_cells=10)
sc.pp.normalize_total(sc_data, target_sum=1e6)
sc.pp.log1p(sc_data)
sc_data.var_names_make_unique()
print(f"  Filtered: {sc_data.n_obs:,} cells x {sc_data.n_vars:,} genes")

print("  [3/5] Cell type distribution...")
cell_type_counts = sc_data.obs['cell_type'].value_counts()
print(f"  Found {len(cell_type_counts)} cell types")
print("  Top 10 cell types:")
for ct, count in cell_type_counts.head(10).items():
    print(f"    {ct}: {count:,} cells")

print("  [4/5] Creating sparse reference matrix...")
unique_cell_types = sc_data.obs['cell_type'].cat.categories if sc_data.obs['cell_type'].dtype.name == 'category' else sc_data.obs['cell_type'].unique()

ref_dict = {}
n_types = len(unique_cell_types)
for i, ct in enumerate(unique_cell_types):
    print(f"    Computing mean for {ct} [{i+1}/{n_types}]", end='\r')
    mask = sc_data.obs['cell_type'] == ct
    ct_expr = sc_data[mask].X
    
    if sparse.issparse(ct_expr):
        ct_mean = np.array(ct_expr.mean(axis=0)).flatten()
    else:
        ct_mean = np.mean(ct_expr, axis=0)
    
    ref_dict[ct] = ct_mean

print("\n  ? All cell type means computed!")

# Create final reference matrix
ref_matrix = np.vstack([ref_dict[ct] for ct in unique_cell_types])
ref_expr = pd.DataFrame(
    ref_matrix,
    index=unique_cell_types,
    columns=sc_data.var_names
)

print("  [5/5] Saving reference...")
joblib.dump({
    'ref_expr': ref_expr,
    'cell_types': ref_expr.index.tolist(),
    'n_cells': sc_data.n_obs,
    'n_genes': sc_data.n_vars,
    'gene_names': sc_data.var_names.tolist()
}, "data/dtd_reference_bone_marrow.joblib")

print(f"\n" + "="*60)
print("SUMMARY:")
print(f"  ? Reference: {ref_expr.shape[0]} cell types x {ref_expr.shape[1]:,} genes")
print(f"  ? File saved: data/dtd_reference_bone_marrow.joblib")
print(f"  ? Top cell types: {ref_expr.index[:5].tolist()}")
print("="*60)
