import pandas as pd
import joblib
from pathlib import Path

# --- PATH CONFIG ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

print("="*60)
print("RESTORE V2: Smart Extraction of BM Reference")
print("="*60)

# 1. Load the Merged File
merged_path = DATA_DIR / "dtd_reference_merged.joblib"
if not merged_path.exists():
    print("❌ CRITICAL: 'dtd_reference_merged.joblib' is missing!")
    exit()

print(f"[1/3] Loading Merged Reference: {merged_path.name}")
merged_data = joblib.load(merged_path)
merged_expr = merged_data['ref_expr']

# 2. Detect Orientation & Filter
print("\n[2/3] analyzing orientation...")
# We assume Cell Types are the dimension with ~50-100 entries, Genes have ~20k
rows, cols = merged_expr.shape
print(f"   Shape: {rows} rows x {cols} cols")

bm_expr = None

# Case A: Cell Types are Rows (Standard for some ML tools)
if rows < cols:
    print("   ✓ Detected: Cell Types are ROWS")
    # Filter Index for 'BM_'
    bm_rows = [i for i in merged_expr.index if str(i).startswith("BM_")]
    if not bm_rows:
        # Fallback if prefix missing
        bm_rows = [i for i in merged_expr.index if not str(i).startswith("Blood_")]
    
    print(f"   ✓ Found {len(bm_rows)} BM cell types")
    bm_expr = merged_expr.loc[bm_rows, :]
    
    # Clean names (Index)
    clean_index = [i.replace("BM_", "") for i in bm_expr.index]
    bm_expr.index = clean_index

# Case B: Cell Types are Columns (Standard for bioinformatics)
else:
    print("   ✓ Detected: Cell Types are COLUMNS")
    # Filter Columns for 'BM_'
    bm_cols = [c for c in merged_expr.columns if str(c).startswith("BM_")]
    if not bm_cols:
        bm_cols = [c for c in merged_expr.columns if not str(c).startswith("Blood_")]
    
    print(f"   ✓ Found {len(bm_cols)} BM cell types")
    bm_expr = merged_expr[bm_cols]
    
    # Clean names (Columns)
    clean_cols = [c.replace("BM_", "") for c in bm_expr.columns]
    bm_expr.columns = clean_cols

# 3. Save Correctly
print("\n[3/3] Saving verified BM reference...")

# Ensure we save the list of CELL TYPES, not Genes
if bm_expr.shape[0] < bm_expr.shape[1]:
    cell_types = list(bm_expr.index)  # Rows are cells
else:
    cell_types = list(bm_expr.columns) # Cols are cells

bm_data = {
    'ref_expr': bm_expr,
    'cell_types': cell_types
}

output_path = DATA_DIR / "dtd_reference_bm_only.joblib"
joblib.dump(bm_data, output_path)

print(f"   ✅ SUCCESS: Saved to {output_path.name}")
print(f"   ✓ Contains {len(cell_types)} cell types (Should be ~26)")