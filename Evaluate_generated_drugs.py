# ------------------------
# Load packages
# ------------------------

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

import model as m
from model import CDR_model

RDLogger.DisableLog("rdApp.*")

# ------------------------
# Configuration
# ------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = OUT_DIR / "Models" / "CDR_model_cold-drug.pt"       # from Train_full_model.py
GENERATED_SMILES_PATH = OUT_DIR / "generated_drugs.csv"

# Which cell line(s) to evaluate the generated compounds against:
#   "all"            -> every cell line the model was trained on (broad-spectrum potency screen)
#   ["ACH-000004"]   -> a specific depmap_id (or a short list), e.g. a cell line you actually care about
TARGET_CELLS = ["ACH-000004"]

BATCH_SIZE = 256   # drug x cell pairs scored per forward pass

device = m.device
use_dti = m.use_dti
use_crispr = m.use_crispr

# ------------------------
# Load generated candidates and features
# ------------------------

gen_df = pd.read_csv(GENERATED_SMILES_PATH)
smiles_list = gen_df["smiles"].astype(str).tolist()
print(f"Loaded {len(smiles_list)} generated candidates from {GENERATED_SMILES_PATH}")

print("Loading pre-computed drug features...")
fps = np.load(OUT_DIR / "generated_drug_fingerprints.npz")["fps"].astype(np.float32)
chem_emb = np.load(OUT_DIR / "generated_drug_chemberta_embeddings.npz")["emb"]
graph_emb = np.load(OUT_DIR / "generated_drug_molecular_graphs.npz")["graph"]

if use_dti:
    print("Loading pre-computed DTI features...")
    dti_emb = np.load(OUT_DIR / "generated_drug_dti.npz")["dti"]

# Calculate QED for summary statistics
print("Calculating QED scores...")
gen_df["qed"] = [QED.qed(Chem.MolFromSmiles(s)) for s in smiles_list]

# ------------------------
# Load the trained CDR model
# ------------------------

net = CDR_model()
ckpt = torch.load(MODEL_PATH, map_location=device)
net.model.load_state_dict(ckpt["model_state_dict"])
net.model.eval()
model = net.model
print(f"Loaded trained model from {MODEL_PATH}")

# ------------------------
# Build the cell-line panel
# ------------------------

if TARGET_CELLS == "all":
    cell_ids = m._expr.index.tolist()
else:
    cell_ids = list(TARGET_CELLS)

print(f"Scoring against {len(cell_ids)} cell line(s)")

# ------------------------
# Score every (generated drug x cell line) pair
# ------------------------

drug_fp_t = torch.from_numpy(fps).to(device)
drug_seq_t = torch.from_numpy(chem_emb).to(device)
drug_graph_t = torch.from_numpy(graph_emb).to(device)

if use_dti:
    drug_dti_t = torch.from_numpy(dti_emb).to(device)

n_drugs = drug_fp_t.shape[0]
results = []

with torch.no_grad():
    for depmap_id in cell_ids:
        # Load cell line features onto device
        v_expr = torch.from_numpy(m._expr.loc[depmap_id].values.astype(np.float32)).to(device)
        v_mut = torch.from_numpy(m._mut.loc[depmap_id].values.astype(np.float32)).to(device)
        v_meth = torch.from_numpy(m._meth.loc[depmap_id].values.astype(np.float32)).to(device)
        v_cn = torch.from_numpy(m._cn.loc[depmap_id].values.astype(np.float32)).to(device)
        
        if use_crispr:
            v_cri = torch.from_numpy(m._crispr.loc[depmap_id].values.astype(np.float32)).to(device)

        for start in range(0, n_drugs, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_drugs)
            b = end - start

            # Build batch list (adjust order here if your model.py expects something different)
            batch = [
                drug_fp_t[start:end],
                drug_seq_t[start:end],
                drug_graph_t[start:end]
            ]
            
            if use_dti:
                batch.append(drug_dti_t[start:end])
                
            batch.extend([
                v_expr.unsqueeze(0).repeat(b, 1),
                v_mut.unsqueeze(0).repeat(b, 1),
                v_meth.unsqueeze(0).repeat(b, 1),
                v_cn.unsqueeze(0).repeat(b, 1),
            ])
            
            if use_crispr:
                batch.append(v_cri.unsqueeze(0).repeat(b, 1))
                
            batch.append(torch.zeros(b, device=device))  # dummy label

            score, _ = model(batch)
            pred = score.squeeze(-1).cpu().numpy()

            for k in range(b):
                results.append({
                    "smiles": smiles_list[start + k],
                    "depmap_id": depmap_id,
                    "pred_lnIC50": float(pred[k]),
                })

results_df = pd.DataFrame(results)

# ------------------------
# Summarise per compound and save
# ------------------------

mean_df = (results_df.groupby("smiles", as_index=False)["pred_lnIC50"]
           .mean().rename(columns={"pred_lnIC50": "mean_pred_lnIC50"}))

idx_min = results_df.groupby("smiles")["pred_lnIC50"].idxmin()
best_df = (results_df.loc[idx_min, ["smiles", "depmap_id", "pred_lnIC50"]]
           .rename(columns={"depmap_id": "most_sensitive_cell", "pred_lnIC50": "min_pred_lnIC50"}))

summary = (mean_df.merge(best_df, on="smiles")
           .merge(gen_df[["smiles", "qed"]], on="smiles")
           .sort_values("mean_pred_lnIC50")
           .reset_index(drop=True))

full_path = OUT_DIR / "generated_drug_predictions_full.csv"
summary_path = OUT_DIR / "generated_drug_predictions_summary.csv"
results_df.to_csv(full_path, index=False)
summary.to_csv(summary_path, index=False)

print("\nTop 10 candidates by mean predicted lnIC50 (lower = predicted more potent):")
print(summary.head(10).to_string(index=False))
print(f"\nSaved every drug x cell pair to: {full_path}")
print(f"Saved per-compound summary to:   {summary_path}")