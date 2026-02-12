import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy import sparse

print("="*60)
print("STEP 2: Process Blood Single-Cell (SPARSE - Memory Efficient)")
print("="*60)

sc_path = Path("data/blood_sc.h5ad")
sc_data = sc.read_h5ad(sc_path)
print(f"[1/6] Loaded: {sc_data.n_obs:,} cells x {sc_data.n_vars:,} genes")

print(f"[2/6] Using cell type column: cell_type")
print(sc_data.obs["cell_type"].value_counts().head(10))

print("[3/6] Filtering genes...")
sc.pp.filter_genes(sc_data, min_cells=10)
print(f"  ? {sc_data.n_vars:,} genes remaining")

print("[4/6] Normalizing...")
sc.pp.normalize_total(sc_data, target_sum=1e6)
sc.pp.log1p(sc_data)
sc_data.var_names_make_unique()
print("  ? Normalized + log-transformed")

print("[5/6] Creating reference matrix (SPARSE)...")
# KEEP SPARSE - average by cell type directly
cell_types_unique = sc_data.obs["cell_type"].cat.categories if sc_data.obs["cell_type"].dtype.name == "category" else sorted(sc_data.obs["cell_type"].unique())

ref_list = []
for ct in cell_types_unique:
    mask = sc_data.obs["cell_type"] == ct
    if mask.sum() > 10:  # Skip rare cell types
        ct_mean = sc_data[mask].X.mean(axis=0).A1 if sparse.issparse(sc_data.X) else sc_data[mask].X.mean(axis=0)
        ref_list.append(pd.Series(ct_mean, index=sc_data.var_names, name=ct))

ref_expr = pd.DataFrame(ref_list).T  # genes x cell types
cell_types = ref_expr.columns.tolist()

print(f"  ? Reference: {len(cell_types)} cell types x {ref_expr.shape[1]:,} genes")

print("[6/6] Saving...")
joblib.dump({
    "ref_expr": ref_expr,
    "cell_types": cell_types,
    "tissue": "blood"
}, "data/dtd_reference_blood.joblib")

print("\n" + "="*60)
print("SUCCESS!")
print(f"  Final reference: {ref_expr.shape[1]} genes x {len(cell_types)} cell types")
print(f"  Saved: data/dtd_reference_blood.joblib")
print("="*60)
