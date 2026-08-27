# ------------------------
# Load packages
# ------------------------

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
import requests
from DeepPurpose import DTI as models
from DeepPurpose import utils
from dgl.nn import AvgPooling
import gc

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"
use_dti = False

# ------------------------
# Load in data
# ------------------------

# Main cell drug response data
all_data = pd.read_csv(PROCESSED_DIR / 'Response_processed.csv')

# Get the unique drugs and their SMILES
drugs = all_data[['drug_name', 'smiles']].drop_duplicates().reset_index(drop=True)
smiles_list = drugs['smiles'].tolist()

# ------------------------
# Fingerprints
# ------------------------

# Define settings 
RADIUS = 2
NBITS = 1024

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=NBITS,
    includeChirality=False,   # matches typical defaults
    useBondTypes=True,
    onlyNonzeroInvariants=False)

fps = np.zeros((len(drugs), NBITS), dtype=np.uint8)
bad_rows = []

# Compute fingerprints for each drug's SMILES
for i, (drug_name, smi) in enumerate(zip(drugs["drug_name"], drugs["smiles"])):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        bad_rows.append({"row": i, "drug_name": str(drug_name), "smiles": str(smi)})
        continue

    fp = morgan_gen.GetFingerprint(mol)  # ExplicitBitVect
    arr = np.zeros((NBITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps[i] = arr

# Make a DataFrame for inspection
fp_cols = [f"fp_{j}" for j in range(fps.shape[1])]
fps_df = pd.DataFrame(fps, columns=fp_cols)
fps_df.insert(0, "smiles", drugs["smiles"].astype(str).values)
fps_df.insert(0, "drug_name", drugs["drug_name"].astype(str).values)

print('Got fingerpint features')

# ------------------------
# ChemBERTa
# ------------------------

# Settings
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
MAX_LEN = 512  # token max length

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

# Compute embeddings (masked mean pooling)
with torch.no_grad():
    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    last = out.last_hidden_state                    # (N, L, H)
    mask = enc["attention_mask"].unsqueeze(-1)      # (N, L, 1)

    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (N, H)
    emb = pooled.detach().cpu().numpy().astype(np.float32)
    
# Make a DataFrame for inspection
emb_cols = [f"chemberta_{i}" for i in range(emb.shape[1])]
chem_df = pd.DataFrame(emb, columns=emb_cols)
chem_df.insert(0, "smiles", drugs["smiles"].astype(str).values)
chem_df.insert(0, "drug_name", drugs["drug_name"].astype(str).values)

print('Got ChemBERTa features')

# ------------------------
# Molecular graphs
# ------------------------

# Settings
GRAPH_MODEL_NAME = "gin_supervised_masking"
device = "cuda" if torch.cuda.is_available() else "cpu"
# Pretrained DGL-LifeSci model + pooling (same pattern as model.py)
model_drug = load_pretrained(GRAPH_MODEL_NAME).to(device).eval()
readout = AvgPooling()
atom_featurizer = PretrainAtomFeaturizer()
bond_featurizer = PretrainBondFeaturizer()

# Compute graph embeddings
N = len(drugs)
graph = None  
bad_rows = []

with torch.no_grad():
    for i, (drug_name, smi) in enumerate(
        tqdm(zip(drugs["drug_name"].astype(str), drugs["smiles"].astype(str)), total=N)):

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad_rows.append({"row": i, "drug_name": drug_name, "smiles": smi})
            continue

        g = mol_to_bigraph(
            mol,
            add_self_loop=True,
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
            canonical_atom_order=False,
        ).to(device)

        # Same feature extraction as TransCDR model.py
        nfeats = [g.ndata.pop("atomic_number"), g.ndata.pop("chirality_type")]
        efeats = [g.edata.pop("bond_type"), g.edata.pop("bond_direction_type")]

        node_repr = model_drug(g, nfeats, efeats)          # (num_nodes, D)
        pooled = readout(g, node_repr)                     # (1, D)
        vec = pooled.detach().cpu().numpy().reshape(-1).astype(np.float32)

        if graph is None:
            D = vec.shape[0]
            graph = np.zeros((N, D), dtype=np.float32)

        graph[i] = vec

graph_cols = [f"graph_{j}" for j in range(graph.shape[1])]
graph_df = pd.DataFrame(graph, columns=graph_cols)
graph_df.insert(0, "smiles", drugs["smiles"].astype(str).values)
graph_df.insert(0, "drug_name", drugs["drug_name"].astype(str).values)

print('Got molecular graph features')

# ------------------------
# Create depmap target list
# ------------------------

if use_dti:
    # Load up the depmap file
    crispr_gene_effect = pd.read_csv( RAW_DIR / 'CRISPRGeneEffect_(RAW).csv', index_col=0)

    # Calculate variance for every gene across all cell lines
    variances = crispr_gene_effect.var()

    # Sort descending to find targets with the most differential essentiality
    top_n_genes = variances.sort_values(ascending=False).head(500)

    # DepMap columns are formatted as "SYMBOL (EntrezID)" - e.g., "EGFR (1956)"
    depmap_genes = [gene.split(" ")[0] for gene in top_n_genes.index]

    # ------------------------
    # Create LINCS L1000 target list
    # ------------------------

    # Load up Lincs file
    lincs = pd.read_csv(RAW_DIR / "GSE92742_Broad_LINCS_gene_info_(RAW).txt", sep='\t')

    # The column 'pr_is_lmk' has a 1 if it's a landmark gene, and a 0 if it isn't
    landmark_df = lincs[lincs['pr_is_lm'] == 1]

    # Extract the gene symbols as a Python list
    lincs_genes = landmark_df['pr_gene_symbol'].tolist()

    # ------------------------
    # Create a union of the two lists
    # ------------------------

    target_genes = set(depmap_genes + lincs_genes)

    # ------------------------
    # Convert to uniprot sequences 
    # ------------------------

    def get_fasta_from_genes(target_genes, batch_size=100):
        fasta_dict = {}
        base_url = "https://rest.uniprot.org/uniprotkb/search"
        
        # Process in batches to avoid URL length limits
        for i in range(0, len(target_genes), batch_size):
            batch = target_genes[i:i+batch_size]
            
            # Query exact human gene symbols that are Swiss-Prot reviewed
            query_parts = [f"gene_exact:{gene}" for gene in batch]
            query_string = "(" + " OR ".join(query_parts) + ") AND organism_id:9606 AND reviewed:true"
            
            params = {
                "query": query_string,
                "format": "json",
                "fields": "gene_names,sequence",
                "size": 500}
            
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                for entry in results:
                    try:
                        # Extract primary gene name and sequence
                        gene_symbol = entry["genes"][0]["geneName"]["value"]
                        sequence = entry["sequence"]["value"]
                        
                        # Verify the hit is in our batch to avoid synonym mismatches
                        if gene_symbol in batch and gene_symbol not in fasta_dict:
                            fasta_dict[gene_symbol] = sequence
                    except KeyError:
                        continue
            else:
                print(f"Failed on batch {i}")
                
        return fasta_dict

    target_dict = get_fasta_from_genes(list(target_genes))
    print('Got sequences for', (len(target_dict)/len(target_genes))*100, '% of genes')

    # ------------------------
    # Build the full drug x target pair list (cross product)
    # ------------------------

    drug_names = drugs['drug_name'].tolist()
    drug_smiles = drugs['smiles'].tolist()

    target_names = list(target_dict.keys())
    target_seqs = list(target_dict.values())

    X_drug = []
    X_target = []
    pair_drug = []
    pair_smiles = []
    pair_target = []

    for d_name, d_smi in zip(drug_names, drug_smiles):
        for t_name, t_seq in zip(target_names, target_seqs):
            X_drug.append(d_smi)
            X_target.append(t_seq)
            pair_drug.append(d_name)
            pair_smiles.append(d_smi)
            pair_target.append(t_name)

    print(f"{len(X_drug):,} drug-target pairs to score")

    # Dummy target values required by DeepPurpose
    y_dummy = [0.0] * len(X_drug)

    # ------------------------
    # Load the locally available pretrained MPNN-CNN model
    # ------------------------

    mpnn_model = models.model_pretrained(
        path_dir=r"C:\Users\benma\Cancer\TransCDR_recreate\save_folder\pretrained_models\mpnn_cnn_bindingdb_ic50"
    )

    print("MPNN-CNN BindingDB IC50 model loaded")

    # ------------------------
    # Convert drug/target pairs into DeepPurpose features
    # ------------------------

    X_pred = utils.data_process(
        X_drug,
        X_target,
        y_dummy,
        'MPNN',
        'CNN',
        split_method='no_split'
    )

    print("DeepPurpose features prepared")

    # ------------------------
    # Predict binding affinity
    # ------------------------

    pred_mpnn = mpnn_model.predict(X_pred)

    print("DTI predictions generated")

    # ------------------------
    # Assemble results
    # ------------------------

    result = pd.DataFrame({
        'drug_name': pair_drug,
        'smiles': pair_smiles,
        'target_gene': pair_target,
        'pred_mpnn': pred_mpnn
    })

    # ------------------------
    # Create drug x target matrix
    # ------------------------

    dti_matrix = result.pivot(
        index='drug_name',
        columns='target_gene',
        values='pred_mpnn'
    )

    # Add smiles back so dti_df has the same basic structure as the other feature DataFrames
    dti_df = dti_matrix.reset_index()

    dti_df = dti_df.merge(
        drugs[['drug_name', 'smiles']].drop_duplicates(),
        on='drug_name',
        how='left')

    # Put columns in the same order as the other feature DataFrames
    target_columns = [col for col in dti_df.columns
                    if col not in ['drug_name', 'smiles']]

    dti_df = dti_df[['drug_name', 'smiles'] + target_columns]
    dti = dti_df.iloc[:, 2:].to_numpy(dtype=np.float32)

    print('Got drug-target interaction features')

# ------------------------
# Save to file
# ------------------------

# Cache for fast load later
np.savez_compressed(PROCESSED_DIR /"drug_fingerprints.npz",fps=fps, drug_name=drugs["drug_name"].astype(str).to_numpy(), smiles=drugs["smiles"].astype(str).to_numpy()) # Fingerprints
np.savez_compressed(PROCESSED_DIR / "drug_chemberta_embeddings.npz", emb=emb, drug_name=drugs["drug_name"].astype(str).to_numpy(), smiles=drugs["smiles"].astype(str).to_numpy()) # ChemBERTa
np.savez_compressed(PROCESSED_DIR / "drug_molecular_graphs.npz", graph=graph, drug_name=drugs["drug_name"].astype(str).to_numpy(), smiles=drugs["smiles"].astype(str).to_numpy()) # Molecular graphs
if use_dti:
    np.savez_compressed(PROCESSED_DIR / "drug_dti.npz", dti=dti, drug_name=drugs["drug_name"].astype(str).to_numpy(), smiles=drugs["smiles"].astype(str).to_numpy()) # Drug-target interaction

print('Saved')