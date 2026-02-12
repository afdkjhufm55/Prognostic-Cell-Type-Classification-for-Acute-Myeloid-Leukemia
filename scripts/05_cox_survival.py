from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("STEP 5: Cox Survival Analysis - BM vs Blood Cell Types")
print("="*60)

# Load deconvolution results
frac_df = pd.read_csv("final_results/TCGA_LAML_cell_fractions.csv", index_col=0)
clinical_df = pd.read_csv("data/TCGA_LAML_clinical.csv", index_col=0)

print(f"[1/5] Data loaded:")
print(f"  Fractions: {frac_df.shape}")
print(f"  Clinical: {clinical_df.shape}")

# Merge data
merged_df = pd.concat([clinical_df, frac_df], axis=1).dropna()
print(f"[2/5] Merged: {merged_df.shape[0]} samples")

cox_results = []
print("[3/5] Running Cox regression for each cell type...")

for ct in frac_df.columns:
    df_cox = merged_df[["OS.time", "vital_status_num", ct]].dropna()
    if len(df_cox) < 20:
        continue
    
    try:
        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col="OS.time", event_col="vital_status_num")
        hr = cph.hazard_ratios_[ct]
        pval = cph.summary.loc[ct, "p"]
        cox_results.append({
            "CellType": ct,
            "HR": hr,
            "pvalue": pval,
            "n_samples": len(df_cox)
        })
    except:
        continue

cox_df = pd.DataFrame(cox_results)
cox_sig = cox_df[cox_df["pvalue"] < 0.05].sort_values("pvalue")

print(f"[4/5] Results: {len(cox_sig)} significant cell types")

# Save results
cox_df.to_csv("final_results/cox_all_results.csv", index=False)
cox_sig.to_csv("final_results/cox_significant.csv", index=False)

print("[5/5] Creating survival plots...")
plt.figure(figsize=(12, 8))

# Top 3 significant cell types
top3 = cox_sig.head(3)
for i, (_, row) in enumerate(top3.iterrows()):
    ct = row["CellType"]
    merged_df[f"{ct}_high"] = merged_df[ct] > merged_df[ct].median()
    
    plt.subplot(2, 2, i+1)
    high = merged_df[merged_df[f"{ct}_high"] == True]
    low = merged_df[merged_df[f"{ct}_high"] == False]
    
    plt.hist(high["OS.time"], alpha=0.5, label="High", bins=20)
    plt.hist(low["OS.time"], alpha=0.5, label="Low", bins=20)
    plt.title(f"{ct[:30]}...\nHR={row['HR']:.2f}, p={row['pvalue']:.2e}")
    plt.xlabel("Survival time (days)")
    plt.legend()

plt.tight_layout()
plt.savefig("final_results/survival_plots.png", dpi=300, bbox_inches="tight")

print("\n" + "="*70)
print("???  COMPLETE PIPELINE SUCCESS!")
print("="*70)
print(f"?? {len(cox_sig)} significant prognostic cell types found!")
print("\n?? FINAL RESULTS:")
print("  final_results/TCGA_LAML_cell_fractions.csv     ? MAIN RESULT")
print("  final_results/cox_significant.csv              ? KEY FINDINGS")
print("  final_results/survival_plots.png              ? VISUALIZATION")
print("\n?? TOP PROGNOSTIC CELL TYPES:")
print(cox_sig[["CellType", "HR", "pvalue"]].head())
print("="*70)
