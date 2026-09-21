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
PROCESSED_DIR = PROJECT_ROOT / "processed"
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
required_columns = {"drug_name", "smiles"}
if missing_columns := required_columns - set(gen_df.columns):
    raise ValueError(f"Generated-drug table is missing columns: {sorted(missing_columns)}")
if gen_df.empty or gen_df[list(required_columns)].isna().any().any():
    raise ValueError("Generated-drug table must contain non-null candidates.")
if gen_df["smiles"].duplicated().any():
    raise ValueError("Generated-drug table contains duplicate SMILES.")

smiles_list = gen_df["smiles"].astype(str).tolist()
print(f"Loaded {len(smiles_list)} generated candidates from {GENERATED_SMILES_PATH}")

def load_feature_cache(path, key):
    with np.load(path, allow_pickle=True) as cache:
        if key not in cache or "smiles" not in cache:
            raise ValueError(f"Malformed feature cache: {path}")
        values = cache[key].astype(np.float32)
        cached_smiles = cache["smiles"].astype(str).tolist()

    if cached_smiles != smiles_list:
        raise ValueError(f"SMILES order in {path.name} does not match {GENERATED_SMILES_PATH.name}.")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Feature array {key!r} in {path.name} must be a finite 2D matrix.")
    return values

print("Loading pre-computed drug features...")
fps = load_feature_cache(OUT_DIR / "generated_drug_fingerprints.npz", "fps")
chem_emb = load_feature_cache(OUT_DIR / "generated_drug_chemberta_embeddings.npz", "emb")
graph_emb = load_feature_cache(OUT_DIR / "generated_drug_molecular_graphs.npz", "graph")
expected_dims = {"fingerprints": m.fp_input_dim, "ChemBERTa": m.seq_input_dim, "graphs": m.graph_input_dim}
for feature_name, values in (("fingerprints", fps), ("ChemBERTa", chem_emb), ("graphs", graph_emb)):
    if values.shape[1] != expected_dims[feature_name]:
        raise ValueError(f"Generated {feature_name} dimension does not match the trained feature cache.")

if use_dti:
    print("Loading pre-computed DTI features...")
    dti_path = OUT_DIR / "generated_drug_dti.npz"
    dti_emb = load_feature_cache(dti_path, "dti")
    if dti_emb.shape[1] != m.dti_input_dim:
        raise ValueError("Generated DTI dimension does not match the trained feature cache.")
    with np.load(dti_path, allow_pickle=True) as generated_dti, np.load(PROCESSED_DIR / "drug_dti.npz", allow_pickle=True) as training_dti:
        if "target_gene" not in generated_dti or "target_gene" not in training_dti:
            raise ValueError("DTI caches must include target_gene metadata.")
        if not np.array_equal(generated_dti["target_gene"].astype(str), training_dti["target_gene"].astype(str)):
            raise ValueError("Generated and training DTI target columns do not match.")

# Calculate QED for summary statistics
print("Calculating QED scores...")
gen_df["qed"] = [QED.qed(Chem.MolFromSmiles(s)) for s in smiles_list]

# ------------------------
# Load the trained CDR model
# ------------------------

ckpt = torch.load(MODEL_PATH, map_location=device)
for flag_name, current_value in (("use_crispr", use_crispr), ("use_dti", use_dti)):
    if flag_name in ckpt and ckpt[flag_name] != current_value:
        raise ValueError(f"Checkpoint {flag_name}={ckpt[flag_name]} but model.py has {flag_name}={current_value}.")

actual_drug_dims = {"fingerprints": fps.shape[1], "chemberta": chem_emb.shape[1], "graphs": graph_emb.shape[1], **({"dti": dti_emb.shape[1]} if use_dti else {})}
for feature_name, expected_dim in ckpt.get("drug_feature_dimensions", {}).items():
    if actual_drug_dims.get(feature_name) != expected_dim:
        raise ValueError(f"Generated {feature_name} features do not match the checkpoint dimension.")

feature_frames = {
    "expression": m._expr,
    "mutation": m._mut,
    "mrna": m._mrna,
    "copy_number": m._cn,
    **({"crispr": m._crispr} if use_crispr else {}),
}
for feature_name, expected_columns in ckpt.get("feature_columns", {}).items():
    if feature_name in feature_frames and expected_columns != feature_frames[feature_name].columns.tolist():
        raise ValueError(f"Current {feature_name} columns do not match the checkpoint schema.")

expression_data = m._expr
if ckpt.get("expression_scaler") is not None:
    scaler = ckpt["expression_scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    if len(mean) != m._expr.shape[1] or len(scale) != m._expr.shape[1]:
        raise ValueError("Checkpoint expression scaler does not match the expression feature matrix.")
    expression_data = pd.DataFrame((m._expr.to_numpy(dtype=np.float32) - mean) / scale, index=m._expr.index, columns=m._expr.columns)

net = CDR_model()
net.model.load_state_dict(ckpt["model_state_dict"])
net.model.eval()
model = net.model
print(f"Loaded trained model from {MODEL_PATH}")

# ------------------------
# Build the cell-line panel
# ------------------------

if TARGET_CELLS == "all":
    cell_ids = m._expr.index.tolist()
elif isinstance(TARGET_CELLS, str):
    cell_ids = [TARGET_CELLS]
else:
    cell_ids = list(TARGET_CELLS)

missing_cells = {cell_id for cell_id in cell_ids for frame in feature_frames.values() if cell_id not in frame.index}
if missing_cells:
    raise KeyError(f"Target cell lines are missing required features: {sorted(missing_cells)}")

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
        v_expr = torch.from_numpy(expression_data.loc[depmap_id].values.astype(np.float32)).to(device)
        v_mut = torch.from_numpy(m._mut.loc[depmap_id].values.astype(np.float32)).to(device)
        v_mrna = torch.from_numpy(m._mrna.loc[depmap_id].values.astype(np.float32)).to(device)
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
                v_mrna.unsqueeze(0).repeat(b, 1),
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
