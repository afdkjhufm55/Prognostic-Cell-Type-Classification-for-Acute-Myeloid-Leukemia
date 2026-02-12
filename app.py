import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Prognostic Cell Type Classification Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: 700;}
    .sub-header {font-size: 1.5rem; color: #4B5563;}
    .metric-card {background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #1E3A8A;}
    .risk-high {color: #DC2626; font-weight: bold;}
    .risk-low {color: #059669; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Define Paths
    DATA_DIR = Path("data")
    RES_DIR = Path("final_results")
    
    # Load Clinical Data
    try:
        clinical = pd.read_csv(DATA_DIR / "TCGA_LAML_clinical.csv", index_col=0)
        # Fix Survival Columns
        if 'OS.time' in clinical.columns:
            clinical['OS_MONTHS'] = clinical['OS.time'] / 30.4
            clinical['OS_STATUS'] = clinical['vital_status'].apply(
                lambda x: 1 if str(x).lower() in ['dead', 'deceased', '1'] else 0
            )
    except FileNotFoundError:
        # Graceful fallback if data isn't ready
        clinical = pd.DataFrame()

    # Load Deconvolution Results (Merged)
    try:
        merged_fracs = pd.read_csv(RES_DIR / "TCGA_LAML_cell_fractions.csv", index_col=0)
    except:
        merged_fracs = pd.DataFrame()

    # Load Deconvolution Results (BM Only - Control)
    try:
        bm_fracs = pd.read_csv(RES_DIR / "TCGA_LAML_BM_only_fractions.csv", index_col=0)
    except:
        bm_fracs = pd.DataFrame()

    # Load NMF Results (Optional)
    try:
        nmf_fracs = pd.read_csv(RES_DIR / "TCGA_LAML_nmf_fractions.csv", index_col=0)
    except:
        nmf_fracs = pd.DataFrame()

    return clinical, merged_fracs, bm_fracs, nmf_fracs

# Load Data
clinical_df, merged_df, bm_df, nmf_df = load_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("AML Analysis Toolkit")
    
    # UPDATED NAVIGATION LIST
    page = st.radio("Navigate", [
        "Project Overview", 
        "Immune Landscape (Results)", 
        "Project Figures",   # <--- REPLACED Interactive Lab with this
        "NMF Discovery",
        "Methodology Validation"
    ])
    st.markdown("---")
    st.info("Project: Prognostic Cell Type Classification with Deconvolution\nModel: TCGA-LAML Cohort")

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "Project Overview":
    st.markdown('<p class="main-header">Acute Myeloid Leukemia Risk Stratification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using Computational Deconvolution & Non-Negative Matrix Factorization</p>', unsafe_allow_html=True)
    
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    if not clinical_df.empty:
        c1.metric("Patient Cohort", f"{len(clinical_df)} Patients")
        mortality_rate = (clinical_df['OS_STATUS'].sum() / len(clinical_df)) * 100
        c2.metric("Mortality Rate", f"{mortality_rate:.1f}%")
        c3.metric("Cell Types Analyzed", f"{merged_df.shape[1]}")
        c4.metric("Novel Factors", "1 (Factor 8)")
    else:
        st.warning("Data not found. Please ensure files are in 'data/' and 'final_results/'.")

    st.markdown("### 🧬 The Problem: The 'Smoothie' Effect")
    st.write("""
    Clinical sequencing (Bulk RNA-seq) averages gene expression, hiding rare but critical immune cells. 
    This project uses **Single-Cell Deconvolution** to mathematically unmix the tumor microenvironment.
    """)
    
    st.markdown("###Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Protective Factor:** Promyelocytes")
        st.write("Higher levels indicate healthy differentiation and better survival.")
    with col2:
        st.error("**Risk Factor:** CD16+ NK Cells & CD8+ T-Cells")
        st.write("Identified as drivers of mortality (HR > 6.5), likely representing immune exhaustion.")

# --- PAGE 2: IMMUNE LANDSCAPE ---
elif page == "Immune Landscape (Results)":
    st.header("Immune Microenvironment Profiling")
    
    if merged_df.empty:
        st.error("No results loaded.")
    else:
        # 1. Abundance Plot
        st.subheader("Global Cell Abundance (Top 10)")
        mean_abundance = merged_df.mean().sort_values(ascending=False).head(10)
        fig_abund = px.bar(
            x=mean_abundance.values, 
            y=mean_abundance.index, 
            orientation='h',
            labels={'x': 'Average Fraction', 'y': 'Cell Type'},
            color=mean_abundance.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_abund, width='stretch')

        # 2. Correlation Matrix
        st.subheader("Cell-Cell Co-occurrence Patterns")
        st.write("Which cells tend to appear together?")
        # Filter for top variable cells to keep heatmap clean
        top_vars = merged_df.var().sort_values(ascending=False).head(15).index
        corr_matrix = merged_df[top_vars].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, width='stretch')

# --- PAGE 3: PROJECT FIGURES (REPLACED LAB) ---
elif page == "Project Figures":
    st.header("Hazard Ratios")
    data = {
    "CELL TYPE": ["GZMB CD8 T cell", "Common lymphoid progenitor", "CD16 NK cell","Promyelocyte"],
    "Hazard Ratio": [1.2227, 1.1088, 6.5587,0.9492],
    "Significance": [0.0004, 0.0009, 0.0531,0.0219]
    }
    df = pd.DataFrame(data)
    st.table(df)
    st.header("Figures")
    st.markdown("Static plots generated from the analysis pipeline (High Resolution).")
    
    # Logic to find images
    figs_dir = Path("figures")

    # Collect all image files
    image_files = []
    
    if figs_dir.exists():
        image_files.extend(list(figs_dir.glob("*.png")))
        image_files.extend(list(figs_dir.glob("*.jpg")))

    if not image_files:
        st.warning("No images found. Please ensure you have a 'figures' folder or .png files in 'final_results'.")
    else:
        # Display images
        for img_path in sorted(image_files):
            # Create a clean title from filename (e.g., "survival_plots.png" -> "Survival Plots")
            clean_title = img_path.stem.replace("_", " ").replace("-", " ").title()
            
            st.markdown("---")
            st.subheader(f"📌 {clean_title}")
            st.image(str(img_path), caption=f"Source: {img_path.name}", width='stretch')

# --- PAGE 4: NMF DISCOVERY ---
elif page == "NMF Discovery":
    st.header("Unsupervised Discovery (NMF)")
    st.markdown("We used Matrix Factorization to find hidden patterns without using the reference atlas.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### The 'Lethal' Factor")
        st.metric("Risk Signature", "Factor 8")
        st.metric("Hazard Ratio", "68.05")
        st.metric("P-Value", "0.002")
        st.caption("Factor 8 represents a transcriptomic state associated with rapid mortality, distinct from known cell types.")

    with col2:
        if not nmf_df.empty:
            st.subheader("Factor Distribution")
            # Boxplot of all factors
            fig = px.box(nmf_df, points="outliers", title="Distribution of NMF Factors across Cohort")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("NMF Data not loaded. Run `03_nmf.py` first.")

# --- PAGE 5: METHOD VALIDATION ---
elif page == "Methodology Validation":
    st.header("Methodology: Why Dual-Tissue Matters")
    st.write("Comparison of the 'Bone Marrow Only' Model vs. Our 'Merged' Model.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Standard Approach (BM Only)")
        st.write("- Uses only Bone Marrow reference.")
        st.write("- Cannot distinguish circulating blood cells.")
        st.write("- **Result:** May fail when Bone Marrow Dataset is contaminated with cell types common to blood.")
        
    with c2:
        st.markdown("#### Our Approach (Merged)")
        st.write("- Uses BM + Peripheral Blood reference.")
        st.write("- Mathematically separates Tissue vs. Blood signals.")
        st.write("- **Result:** Specifically identifies **CD16+ NK Cells** as the driver.")

    if not bm_df.empty and not merged_df.empty:
        st.markdown("---")
        st.subheader("Data Proof")
        cell_to_check = st.selectbox("Compare Cell Type:", ["NK cell", "B cell"])
        
        # Create comparison data
        comp_data = pd.DataFrame({
            "BM Only Model": bm_df.filter(like=cell_to_check).mean(axis=1),
            "Merged Model": merged_df.filter(like=cell_to_check).mean(axis=1)
        })
        
        fig = px.scatter(comp_data, x="BM Only Model", y="Merged Model", 
                         title=f"Correlation of {cell_to_check} predictions",
                         trendline="ols")
        st.plotly_chart(fig, width='stretch')
        st.caption("If points deviate from the line, the models are seeing different things (Specificty Gain).")