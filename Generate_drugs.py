import torch
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from Get_drug_features import compute_morgan_fingerprints, compute_chemberta_embeddings, compute_graph_embeddings, compute_dti_features

RDLogger.DisableLog("rdApp.*")

# ------------------------
# Configuration
# ------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_MODEL_NAME = "entropy/gpt2_zinc_87m"   # GPT-2 (87M params) trained on ~480M ZINC SMILES
N_TO_SAMPLE = 20000                         # total raw sequences to draw
MAX_LEN = 150                              # max token length per SMILES
TEMPERATURE = 1.0                          # >1 = more diverse/riskier, <1 = safer/more repetitive
TOP_K = 50
GEN_BATCH = 128                            # sequences generated per forward pass
USE_DTI = False                            # Set to True if you want DTI feature calculation

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load pretrained de novo generator
# ------------------------

tokenizer = GPT2TokenizerFast.from_pretrained(GEN_MODEL_NAME, max_len=256)
gen_model = GPT2LMHeadModel.from_pretrained(GEN_MODEL_NAME).to(device).eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------
# Sample candidate SMILES
# ------------------------

raw_smiles = []
seed = torch.tensor([[tokenizer.bos_token_id]], device=device)

with torch.no_grad():
    n_batches = (N_TO_SAMPLE + GEN_BATCH - 1) // GEN_BATCH
    for b in range(n_batches):
        gen = gen_model.generate(
            seed,
            do_sample=True,
            max_length=MAX_LEN,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=GEN_BATCH,
        )
        raw_smiles.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

raw_smiles = raw_smiles[:N_TO_SAMPLE]

# ------------------------
# Validate, canonicalise, dedupe, drop anything already in the training set
# ------------------------

known_smiles = set(pd.read_csv(PROCESSED_DIR / "Response_processed.csv")["smiles"].astype(str))

seen, valid_smiles = set(), []
for smi in raw_smiles:
    smi = smi.strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    canon = Chem.MolToSmiles(mol)
    if canon in seen or canon in known_smiles:
        continue
    seen.add(canon)
    valid_smiles.append(canon)

validity = len(valid_smiles) / max(len(raw_smiles), 1)
print(f"Sampled {len(raw_smiles)} raw sequences -> {len(valid_smiles)} valid, novel, unique SMILES "
      f"({validity:.1%} of raw output)")

# ------------------------
# Prepare DataFrame for Feature Calculation
# ------------------------

gen_drugs_df = pd.DataFrame({
    "drug_name": [f"gen_drug_{i+1:04d}" for i in range(len(valid_smiles))],
    "smiles": valid_smiles
})

# Save generated SMILES metadata
gen_drugs_df.to_csv(OUT_DIR / "generated_drugs.csv", index=False)
print(f"Saved generated drug SMILES to {OUT_DIR / 'generated_drugs.csv'}")

# ------------------------
# Compute Features for Generated SMILES
# ------------------------

print("Calculating Morgan Fingerprints...")
fps_df = compute_morgan_fingerprints(gen_drugs_df)
np.savez_compressed(
    OUT_DIR / "generated_drug_fingerprints.npz",
    fps=fps_df.iloc[:, 2:].to_numpy(dtype=np.uint8),
    drug_name=fps_df["drug_name"].to_numpy(),
    smiles=fps_df["smiles"].to_numpy()
)

print("Calculating ChemBERTa Embeddings...")
chem_df = compute_chemberta_embeddings(gen_drugs_df, device=device)
np.savez_compressed(
    OUT_DIR / "generated_drug_chemberta_embeddings.npz",
    emb=chem_df.iloc[:, 2:].to_numpy(dtype=np.float32),
    drug_name=chem_df["drug_name"].to_numpy(),
    smiles=chem_df["smiles"].to_numpy()
)

print("Calculating Molecular Graph Features...")
graph_df = compute_graph_embeddings(gen_drugs_df, device=device)
np.savez_compressed(
    OUT_DIR / "generated_drug_molecular_graphs.npz",
    graph=graph_df.iloc[:, 2:].to_numpy(dtype=np.float32),
    drug_name=graph_df["drug_name"].to_numpy(),
    smiles=graph_df["smiles"].to_numpy()
)

if USE_DTI:
    print("Calculating Drug-Target Interactions...")
    dti_model_path = PROJECT_ROOT / "save_folder" / "pretrained_models" / "mpnn_cnn_bindingdb_ic50"
    dti_df = compute_dti_features(gen_drugs_df, RAW_DIR, dti_model_path)
    np.savez_compressed(
        OUT_DIR / "generated_drug_dti.npz",
        dti=dti_df.iloc[:, 2:].to_numpy(dtype=np.float32),
        drug_name=dti_df["drug_name"].to_numpy(),
        smiles=dti_df["smiles"].to_numpy()
    )

print("All generated drug features computed and saved to:", OUT_DIR)
