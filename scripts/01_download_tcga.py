import pandas as pd
import requests
import json
import io
import gzip
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

print("="*60)
print("STEP 1: DOWNLOAD FULL TCGA-LAML DATA (NO FILTERING)")
print("="*60)

# 1. Query GDC API for File IDs
# ---------------------------------------------------------
print("[1/4] Querying GDC Database...")
files_endpt = "https://api.gdc.cancer.gov/files"

filters = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LAML"]}},
        {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}}
    ]
}

params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,cases.case_id,cases.submitter_id",
    "format": "JSON",
    "size": "500"
}

response = requests.post(files_endpt, headers={"Content-Type": "application/json"}, json=params)
file_data = response.json()["data"]["hits"]
print(f"   ✓ Found {len(file_data)} files to download.")

# 2. Download and Merge Loop
# ---------------------------------------------------------
print("\n[2/4] Downloading & Merging (This preserves ALL genes)...")

all_counts = {}
data_endpt = "https://api.gdc.cancer.gov/data"

for i, info in enumerate(file_data):
    file_id = info["file_id"]
    # Prefer UUID (case_id) but fallback to Barcode (submitter_id)
    patient_id = info["cases"][0]["case_id"] 
    
    if i % 10 == 0:
        print(f"   Processing {i}/{len(file_data)}: {patient_id}...", end='\r')
        
    try:
        # Fetch file content
        file_resp = requests.get(f"{data_endpt}/{file_id}")
        
        # Determine format (Gzipped or Plain)
        try:
            content = gzip.decompress(file_resp.content).decode("utf-8")
        except:
            content = file_resp.content.decode("utf-8")
            
        # Parse TSV (The format is: gene_id  gene_name  type  unstranded ...)
        # We only want 'gene_id' (col 0) and 'unstranded' (col 3)
        df = pd.read_csv(io.StringIO(content), sep="\t", comment="#", header=0)
        
        # CLEANUP: STAR Counts usually have specific columns.
        # We need 'gene_id' as index and 'unstranded' (or 'tpm_unstranded') as value.
        # Check column names:
        if "gene_id" in df.columns and "unstranded" in df.columns:
            # Filter out non-gene stats (N_unmapped, etc.)
            df = df[df["gene_id"].str.startswith("ENSG")]
            all_counts[patient_id] = df.set_index("gene_id")["unstranded"]
            
    except Exception as e:
        print(f"   ⚠️ Error with {patient_id}: {e}")

# 3. Create Master Matrix
# ---------------------------------------------------------
print("\n\n[3/4] Assembling Master Matrix...")
counts_df = pd.DataFrame(all_counts)
print(f"   ✓ Raw Shape: {counts_df.shape[0]} genes x {counts_df.shape[1]} patients")

# 4. Verify Critical Markers
# ---------------------------------------------------------
print("\n[4/4] Verifying Critical Markers...")
markers = ["ENSG00000203747", "ENSG00000162747", "ENSG00000170458"] # FCGR3A, FCGR3B, CD14
missing = [m for m in markers if m not in counts_df.index]

if len(missing) == 0:
    print("   ✅ SUCCESS: All CD16 markers are present!")
else:
    print(f"   ❌ WARNING: Still missing {missing}")
    # Try stripping versions if mismatch
    clean_index = [x.split('.')[0] for x in counts_df.index]
    counts_df.index = clean_index
    missing_clean = [m for m in markers if m not in counts_df.index]
    if len(missing_clean) == 0:
        print("   ✅ SUCCESS: Found markers after ID cleanup!")
    else:
        print("   ❌ CRITICAL FAILURE: Markers truly absent.")

# Save
output_path = DATA_DIR / "TCGA_LAML_counts.csv"
counts_df.to_csv(output_path)
print(f"   ✓ Saved to: {output_path}")