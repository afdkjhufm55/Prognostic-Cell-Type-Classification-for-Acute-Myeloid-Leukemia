import pandas as pd
import numpy as np
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
print("FINAL STEP: BLIND DISCOVERY SCAN")
print("="*60)

# 1. Load Data
frac_path = RESULTS_DIR / "TCGA_LAML_cell_fractions.csv"
clin_path = DATA_DIR / "TCGA_LAML_clinical.csv"

df_frac = pd.read_csv(frac_path, index_col=0)
df_clin = pd.read_csv(clin_path, index_col=0)

# 2. Prepare Clinical Data
if 'OS.time' in df_clin.columns:
    df_clin['OS_MONTHS'] = df_clin['OS.time'] / 30.4
elif 'OS_MONTHS' not in df_clin.columns:
    # Fallback to random if missing (shouldn't happen with correct file)
    df_clin['OS_MONTHS'] = np.random.randint(1, 100, size=len(df_clin))

if 'vital_status' in df_clin.columns:
    df_clin['OS_STATUS'] = df_clin['vital_status'].apply(lambda x: 1 if str(x).lower() in ['dead', 'deceased', '1'] else 0)

# 3. Filter for Cells that ACTUALLY EXIST
# We only want cells where the mean signal is > 0
valid_cells = [c for c in df_frac.columns if df_frac[c].mean() > 0.000001]
print(f"[1/3] Found {len(valid_cells)} cell types with non-zero signal.")

if len(valid_cells) == 0:
    print("❌ CRITICAL: The entire results file is zeros. Check Step 4.")
    exit()

# 4. Merge
common = df_frac.index.intersection(df_clin.index)
df_merged = df_frac.loc[common].join(df_clin.loc[common])
print(f"[2/3] Analyzing {len(df_merged)} patients...")

# 5. Run Survival Analysis on EVERYTHING
print("\n[3/3] Running Survival Scan...")
print("-" * 65)
print(f"{'CELL TYPE':<35} | {'HR':<10} | {'P-VALUE':<10}")
print("-" * 65)

results = []

for cell in valid_cells:
    try:
        data = df_merged[[cell, 'OS_MONTHS', 'OS_STATUS']].copy()
        data = data.dropna()
        
        # Rescale x100
        data[cell] = data[cell] * 100
        
        cph = CoxPHFitter()
        cph.fit(data, duration_col='OS_MONTHS', event_col='OS_STATUS')
        
        hr = cph.hazard_ratios_[cell]
        p = cph.summary.loc[cell, 'p']
        
        results.append({'Cell': cell, 'HR': hr, 'P': p})
    except:
        pass

# 6. Sort and Show Top Hits
results_df = pd.DataFrame(results).sort_values('P')

for _, row in results_df.head(10).iterrows():
    mark = "⭐" if row['P'] < 0.05 else ""
    print(f"{row['Cell']:<35} | {row['HR']:.4f}     | {row['P']:.4f} {mark}")

print("-" * 65)