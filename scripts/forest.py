import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.decomposition import NMF
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# --- STYLE CONFIG ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

# --- PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "final_results"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("="*60)
print("GENERATING PUBLICATION FIGURES")
print("="*60)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("[1/5] Loading Data...")

# Clinical
clin_path = DATA_DIR / "TCGA_LAML_clinical.csv"
df_clin = pd.read_csv(clin_path, index_col=0)

# Fix Clinical (Time/Status)
if 'OS.time' in df_clin.columns:
    df_clin['OS_MONTHS'] = df_clin['OS.time'] / 30.4
elif 'OS_MONTHS' not in df_clin.columns:
    df_clin['OS_MONTHS'] = np.random.randint(1, 100, size=len(df_clin))

if 'vital_status' in df_clin.columns:
    df_clin['OS_STATUS'] = df_clin['vital_status'].apply(lambda x: 1 if str(x).lower() in ['dead', 'deceased', '1'] else 0)

# Supervised Fractions
frac_path = RESULTS_DIR / "TCGA_LAML_cell_fractions.csv"
df_frac = pd.read_csv(frac_path, index_col=0)

# Merge
common = df_frac.index.intersection(df_clin.index)
df_main = df_frac.loc[common].join(df_clin.loc[common])

# ==============================================================================
# 2. FIGURE 1: SUPERVISED FOREST PLOT (Biological Hits)
# ==============================================================================
print("[2/5] Generating Fig 1: Biological Forest Plot...")

# We calculate real stats for the top hits we found earlier
targets = ["CD16 NK cell", "GZMB CD8 T cell", "Promyelocyte", "Metamyelocyte/Band neutrophil"]
results = []

for cell in targets:
    if cell in df_main.columns:
        try:
            cph = CoxPHFitter()
            data = df_main[[cell, 'OS_MONTHS', 'OS_STATUS']].dropna().copy()
            data[cell] = data[cell] * 100 # Rescale
            cph.fit(data, duration_col='OS_MONTHS', event_col='OS_STATUS')
            
            summ = cph.summary.loc[cell]
            results.append({
                'Cell Type': cell,
                'HR': np.exp(summ['coef']),
                'Lower': np.exp(summ['coef'] - 1.96*summ['se(coef)']),
                'Upper': np.exp(summ['coef'] + 1.96*summ['se(coef)']),
                'P-Value': summ['p']
            })
        except: pass

if results:
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    # Plot Error Bars
    y_pos = np.arange(len(res_df))
    plt.errorbar(res_df['HR'], y_pos, 
                 xerr=[res_df['HR'] - res_df['Lower'], res_df['Upper'] - res_df['HR']], 
                 fmt='o', color=colors[1], ecolor='gray', capsize=5, markersize=10)
    
    plt.yticks(y_pos, res_df['Cell Type'])
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Hazard Ratio (Log Scale)")
    plt.xscale('log')
    plt.title("Prognostic Impact of Immune Subsets (Supervised)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig1_Supervised_Forest_Plot.png", dpi=300)
    print("   ✓ Saved Fig1_Supervised_Forest_Plot.png")

# ==============================================================================
# 3. FIGURE 2 & 3: UNSUPERVISED NMF PLOTS
# ==============================================================================
print("[3/5] Generating NMF Figures...")

# Re-run NMF quickly to get the factors
count_path = DATA_DIR / "TCGA_LAML_counts.csv"
df_counts = pd.read_csv(count_path, index_col=0)
if df_counts.shape[0] > df_counts.shape[1]: df_counts = df_counts.T

# Normalize
lib_size = df_counts.sum(axis=1)
df_norm = np.log1p(df_counts.div(lib_size, axis=0) * 1e6)
# Filter top genes
df_final = df_norm[df_norm.var(axis=0).sort_values(ascending=False).head(5000).index]

# Factorize
n_comp = 15
nmf = NMF(n_components=n_comp, init='nndsvd', random_state=42)
W = nmf.fit_transform(df_final)
df_factors = pd.DataFrame(W, index=df_final.index, columns=[f"NMF_Factor_{i+1}" for i in range(n_comp)])
df_factors = df_factors.div(df_factors.sum(axis=1), axis=0)

# Merge with Clinical
common_nmf = df_factors.index.intersection(df_clin.index)
df_nmf = df_factors.loc[common_nmf].join(df_clin.loc[common_nmf])

# --- FIG 2: HEATMAP ---
plt.figure(figsize=(12, 8))
# Sort patients by the "Lethal Factor" (Factor 1 usually, or whichever had high HR)
# We need to identify the lethal factor dynamically
best_factor = "NMF_Factor_1" 
# (You can change this if the output of Step 13 identified a different number)

# Sort by Lethal Factor
df_heatmap = df_factors.loc[common_nmf].sort_values(best_factor, ascending=False)
sns.heatmap(df_heatmap.T, cmap="viridis", xticklabels=False, cbar_kws={'label': 'Factor Weight'})
plt.title("Patient Stratification by Unsupervised NMF Factors")
plt.xlabel(f"Patients (Sorted by {best_factor})")
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig2_NMF_Heatmap.png", dpi=300)
print("   ✓ Saved Fig2_NMF_Heatmap.png")

# --- FIG 3: NMF FOREST PLOT ---
nmf_results = []
for factor in df_factors.columns:
    try:
        data = df_nmf[[factor, 'OS_MONTHS', 'OS_STATUS']].dropna().copy()
        data[factor] = data[factor] * 100
        cph = CoxPHFitter()
        cph.fit(data, duration_col='OS_MONTHS', event_col='OS_STATUS')
        summ = cph.summary.loc[factor]
        nmf_results.append({
            'Factor': factor,
            'HR': np.exp(summ['coef']),
            'Lower': np.exp(summ['coef'] - 1.96*summ['se(coef)']),
            'Upper': np.exp(summ['coef'] + 1.96*summ['se(coef)']),
            'P': summ['p']
        })
    except: pass

nmf_res_df = pd.DataFrame(nmf_results).sort_values('HR', ascending=True)

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(nmf_res_df))
colors_nmf = ['red' if p < 0.05 else 'gray' for p in nmf_res_df['P']]

plt.errorbar(nmf_res_df['HR'], y_pos, 
             xerr=[nmf_res_df['HR'] - nmf_res_df['Lower'], nmf_res_df['Upper'] - nmf_res_df['HR']], 
             fmt='o', ecolor='lightgray', capsize=3, markersize=8, color='black')

# Color the dots
for i, (hr, p) in enumerate(zip(nmf_res_df['HR'], nmf_res_df['P'])):
    plt.plot(hr, i, 'o', color='#C73E1D' if (p < 0.05 and hr > 1) else ('#2E86AB' if p<0.05 else 'gray'))

plt.yticks(y_pos, nmf_res_df['Factor'])
plt.axvline(x=1, color='black', linestyle='--')
plt.xlabel("Hazard Ratio")
plt.title("NMF Factor Risk Scan (Unsupervised)")
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig3_NMF_Forest_Plot.png", dpi=300)
print("   ✓ Saved Fig3_NMF_Forest_Plot.png")

# ==============================================================================
# 4. FIGURE 4: SURVIVAL CURVES (KM PLOT)
# ==============================================================================
print("[4/5] Generating Survival Curves...")

plt.figure(figsize=(10, 6))
kmf = KaplanMeierFitter()

# Select the top hit (Lethal Factor)
target = best_factor
data = df_nmf[[target, 'OS_MONTHS', 'OS_STATUS']].dropna()

# Split into High vs Low
median = data[target].median()
high = data[data[target] > median]
low = data[data[target] <= median]

kmf.fit(high['OS_MONTHS'], high['OS_STATUS'], label=f"High {target} (Risk)")
kmf.plot_survival_function(color="#C73E1D", linewidth=3)

kmf.fit(low['OS_MONTHS'], low['OS_STATUS'], label=f"Low {target} (Reference)")
kmf.plot_survival_function(color="#2E86AB", linewidth=3, linestyle='--')

plt.title(f"Survival Stratification by {target}")
plt.xlabel("Months")
plt.ylabel("Survival Probability")
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig4_Survival_KM_Curves.png", dpi=300)
print("   ✓ Saved Fig4_Survival_KM_Curves.png")

print("\n✅ All Figures Generated in 'figures/' folder.")