from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# --- PATH FIXER START ---
# This ensures the script finds 'final_results' no matter where you run it from
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "final_results"
# --- PATH FIXER END ---

print("="*70)
print("COMPARISON: BM-Only vs BM+Blood Deconvolution + Survival (REAL DATA)")
print("="*70)

# 1. Load BOTH results (Using Fixed Paths)
# ---------------------------------------------------------
print("[1/4] Loading Datasets...")

bm_path = RESULTS_DIR / "TCGA_LAML_BM_only_fractions.csv"
merged_path = RESULTS_DIR / "TCGA_LAML_cell_fractions.csv"
clin_path = DATA_DIR / "TCGA_LAML_clinical.csv"

# Check existence
if not bm_path.exists():
    print(f"❌ Missing: {bm_path}")
    exit()
if not merged_path.exists():
    print(f"❌ Missing: {merged_path}")
    exit()

bm_frac = pd.read_csv(bm_path, index_col=0)
merged_frac = pd.read_csv(merged_path, index_col=0)
clinical = pd.read_csv(clin_path, index_col=0)

print(f"   BM-only Matrix:  {bm_frac.shape} (n={bm_frac.shape[0]})")
print(f"   BM+Blood Matrix: {merged_frac.shape} (n={merged_frac.shape[0]})")
print(f"   Clinical Data:   {clinical.shape} (n={clinical.shape[0]})")

# 2. Survival Analysis Helper
# ---------------------------------------------------------
def run_cox_screening(fraction_df, clinical_df):
    results = []
    # Merge clinical data (Inner join on Index)
    merged = pd.concat([clinical_df, fraction_df], axis=1, join="inner")
    
    if merged.empty:
        print("   ⚠️ WARNING: No common patients between Clinical and Deconvolution data!")
        print("      Check if IDs match (e.g. UUIDs vs Barcodes).")
        return pd.DataFrame()

    # Run Cox for each cell type
    for cell_type in fraction_df.columns:
        # Skip if cell type is rare (all zeros) or constant
        if merged[cell_type].std() == 0:
            continue
            
        data = merged[["OS.time", "vital_status_num", cell_type]].dropna()
        
        try:
            cph = CoxPHFitter()
            cph.fit(data, duration_col="OS.time", event_col="vital_status_num")
            
            hr = cph.hazard_ratios_[cell_type]
            p_val = cph.summary.loc[cell_type, "p"]
            
            results.append({
                "CellType": cell_type,
                "HR": hr,
                "pvalue": p_val
            })
        except:
            continue
            
    if not results:
        return pd.DataFrame()
        
    return pd.DataFrame(results).sort_values("pvalue")

# 3. Analyze BM-Only
# ---------------------------------------------------------
print("\n[2/4] Analyzing BONE MARROW ONLY Model...")
bm_results = run_cox_screening(bm_frac, clinical)

if not bm_results.empty:
    bm_sig = bm_results[bm_results["pvalue"] < 0.05]
    print(f"   ✓ Tested {len(bm_results)} cell types")
    print(f"   ✓ Significant Findings (p<0.05): {len(bm_sig)}")
    if len(bm_sig) > 0:
        print("\n   🏆 TOP HITS (BM ONLY):")
        print(bm_sig.head(5).to_string(index=False))
        bm_sig.to_csv(RESULTS_DIR / "cox_bm_only_significant.csv", index=False)
else:
    print("   ❌ Analysis failed (Empty intersection or convergence error).")
    bm_sig = pd.DataFrame()

# 4. Analyze Merged
# ---------------------------------------------------------
print("\n[3/4] Analyzing MERGED (BM+Blood) Model...")
merged_results = run_cox_screening(merged_frac, clinical)

if not merged_results.empty:
    merged_sig = merged_results[merged_results["pvalue"] < 0.05]
    print(f"   ✓ Tested {len(merged_results)} cell types")
    print(f"   ✓ Significant Findings (p<0.05): {len(merged_sig)}")
    if len(merged_sig) > 0:
        print("\n   🏆 TOP HITS (MERGED):")
        print(merged_sig.head(5).to_string(index=False))
else:
    print("   ❌ Analysis failed.")

# 5. Final Comparison
# ---------------------------------------------------------
print("\n[4/4] Generating Comparison Summary...")
target = "BM_CD16 monocyte"

def get_stats(df, name):
    if df.empty: return f"{name}: No Data"
    row = df[df["CellType"] == target]
    if not row.empty:
        return f"{name}: HR = {row.iloc[0]['HR']:.2f} | p = {row.iloc[0]['pvalue']:.4f}"
    return f"{name}: Not prognostic"

print(f"\n   🔍 Target Check ('{target}'):")
print("      " + get_stats(bm_results, "BM-Only "))
print("      " + get_stats(merged_results, "Merged  "))

print("\n" + "="*70)
print("If HR is > 3.0, you have reproduced the discovery!")