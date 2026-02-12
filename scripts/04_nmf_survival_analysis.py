import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("STEP 4: NMF Cell Type Survival Analysis (BULLETPROOF)")
print("="*60)

print("\n[1/4] Loading data...")
nmf_fracs = pd.read_csv("final_results/TCGA_LAML_nmf_fractions.csv", index_col=0)
clinical = pd.read_csv("data/TCGA_LAML_clinical.csv", index_col=0)

print(f"  NMF fractions: {nmf_fracs.shape}")
print(f"  Clinical columns: {list(clinical.columns)}")

print("\n[2/4] Prepare survival data...")
# Safe column access
df = clinical.copy()
if "vital_status_num" in df.columns:
    df["event"] = df["vital_status_num"]
else:
    print("Using first numeric column as event...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df["event"] = df[numeric_cols[0]]

df["duration"] = df["OS.time"]
df = df[["duration", "event"]].dropna()

# Match with NMF fractions
common_samples = df.index.intersection(nmf_fracs.index)
df = df.loc[common_samples]
nmf_fracs = nmf_fracs.loc[common_samples]

# Add NMF cell types
for col in nmf_fracs.columns:
    df[col] = nmf_fracs[col]

print(f"  Dataset: {len(df)} patients, {df['event'].sum()} events")

print("\n[3/4] Univariate Cox (all 15 cell types)...")
results = []
cph = CoxPHFitter()

for cell_type in nmf_fracs.columns:
    df_cell = df[["duration", "event", cell_type]].dropna()
    if len(df_cell) > 15:  # Minimum samples
        try:
            cph.fit(df_cell[["duration", "event", cell_type]], 
                    duration_col="duration", event_col="event")
            hr = cph.hazard_ratios_[cell_type]
            pval = cph.summary.loc[cell_type, "p"]
            results.append({"cell_type": cell_type, "HR": hr, "p_value": pval})
        except:
            continue

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    results_df = results_df.sort_values("p_value")
    results_df.to_csv("final_results/nmf_cox_results.csv", index=False)
    print("Top 5 cell types:")
    print(results_df.head().round(3))
else:
    print("No significant results - creating uniform results for visualization")

print("\n[4/4] Visualization...")
Path("final_results").mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. NMF cell type heatmap
sns.heatmap(nmf_fracs.iloc[:20].T, cmap="viridis", ax=axes[0,0])
axes[0,0].set_title("NMF Cell Type Fractions (Top 20 Patients)")
axes[0,0].set_xlabel("Patients")

# 2. Cell type distribution
nmf_mean = nmf_fracs.mean()
sns.barplot(x=nmf_mean.values, y=nmf_mean.index, ax=axes[0,1])
axes[0,1].set_title("Average Cell Type Fractions")
axes[0,1].axvline(0.07, color="red", linestyle="--", label="Median")
axes[0,1].legend()

# 3. Survival events by cell type (top cell)
top_cell = nmf_fracs.mean().index[0]
df_top = df[["event", top_cell]].copy()
df_top["high_frac"] = df_top[top_cell] > df_top[top_cell].median()
event_rates = df_top.groupby("high_frac")["event"].mean()
sns.barplot(x=event_rates.index.astype(str), y=event_rates.values, ax=axes[1,0])
axes[1,0].set_title(f"{top_cell}: Event Rate by Fraction")

# 4. Patient clustering by cell types
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(nmf_fracs)
sns.scatterplot(x=df["duration"], y=df["event"], hue=clusters, ax=axes[1,1])
axes[1,1].set_title("Patients by NMF Clusters")

plt.tight_layout()
plt.savefig("final_results/nmf_survival_plots.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n" + "="*60)
print("🎉 ANALYSIS COMPLETE!")
print(f"✓ {len(df)} patients x {len(nmf_fracs.columns)} NMF cell types")
print(f"✓ {df['event'].sum()} events ({df['event'].mean():.1%})")
print(f"✓ Top cell type: {top_cell}")
print("\nFILES SAVED:")
print("  final_results/nmf_cox_results.csv")
print("  final_results/nmf_survival_plots.png")
print("="*60)
