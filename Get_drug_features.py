from pathlib import Path
import requests
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel
import dgl
from dgl.nn import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer

CHEMBERTA_REVISION = "761d6a18cf99db371e0b43baf3e2d21b3e865a20"

def _validate_drugs_df(drugs_df: pd.DataFrame) -> None:
    required = {"drug_name", "smiles"}
    missing = required - set(drugs_df.columns)
    if missing:
        raise ValueError(f"Missing required drug columns: {sorted(missing)}")
    if drugs_df.empty:
        raise ValueError("No drugs were provided for feature calculation.")
    if drugs_df[list(required)].isna().any().any():
        raise ValueError("Drug names and SMILES must not contain missing values.")

# ------------------------
# 1. Morgan Fingerprints
# ------------------------

def compute_morgan_fingerprints(
    drugs_df: pd.DataFrame, 
    radius: int = 2, 
    nbits: int = 1024
) -> pd.DataFrame:
    """
    Computes Morgan Fingerprints for a DataFrame of drugs.
    
    Parameters:
        drugs_df: DataFrame containing 'drug_name' and 'smiles' columns.
        radius: Morgan fingerprint radius.
        nbits: Bit length of the fingerprint.
        
    Returns:
        pd.DataFrame: Contains ['drug_name', 'smiles', 'fp_0', 'fp_1', ...]
    """
    _validate_drugs_df(drugs_df)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nbits,
        includeChirality=False,
        useBondTypes=True,
        onlyNonzeroInvariants=False
    )

    fps = np.zeros((len(drugs_df), nbits), dtype=np.uint8)
    bad_rows = []

    for i, (drug_name, smi) in enumerate(zip(drugs_df["drug_name"], drugs_df["smiles"])):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            bad_rows.append({"row": i, "drug_name": str(drug_name), "smiles": str(smi)})
            continue

        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr

    if bad_rows:
        raise ValueError(f"Invalid SMILES encountered: {bad_rows[:5]}")

    fp_cols = [f"fp_{j}" for j in range(fps.shape[1])]
    fps_df = pd.DataFrame(fps, columns=fp_cols)
    fps_df.insert(0, "smiles", drugs_df["smiles"].astype(str).values)
    fps_df.insert(0, "drug_name", drugs_df["drug_name"].astype(str).values)

    return fps_df

# ------------------------
# 2. ChemBERTa Embeddings
# ------------------------

def compute_chemberta_embeddings(
    drugs_df: pd.DataFrame, 
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1", 
    model_revision: str = CHEMBERTA_REVISION,
    max_len: int = 512,
    batch_size: int = 64,
    device: str = None
) -> pd.DataFrame:
    """
    Computes masked mean-pooled ChemBERTa embeddings for SMILES strings.
    
    Parameters:
        drugs_df: DataFrame containing 'drug_name' and 'smiles' columns.
        model_name: HuggingFace model path/identifier.
        max_len: Maximum token length.
        batch_size: Batch size for model inference to optimize memory.
        device: 'cuda' or 'cpu'. Automatically detected if None.
        
    Returns:
        pd.DataFrame: Contains ['drug_name', 'smiles', 'chemberta_0', ...]
    """
    _validate_drugs_df(drugs_df)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    model = AutoModel.from_pretrained(model_name, revision=model_revision).to(device).eval()

    smiles_list = drugs_df['smiles'].astype(str).tolist()
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i : i + batch_size]
            enc = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            last = out.last_hidden_state                   # (N, L, H)
            mask = enc["attention_mask"].unsqueeze(-1)     # (N, L, 1)

            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embs.append(pooled.detach().cpu().numpy().astype(np.float32))

    emb = np.vstack(all_embs)

    emb_cols = [f"chemberta_{i}" for i in range(emb.shape[1])]
    chem_df = pd.DataFrame(emb, columns=emb_cols)
    chem_df.insert(0, "smiles", drugs_df["smiles"].astype(str).values)
    chem_df.insert(0, "drug_name", drugs_df["drug_name"].astype(str).values)

    return chem_df

# ------------------------
# 3. Molecular Graph Features
# ------------------------

def compute_graph_embeddings(
    drugs_df: pd.DataFrame, 
    model_name: str = "gin_supervised_masking",
    device: str = None
) -> pd.DataFrame:
    """
    Computes graph representation embeddings using a pretrained DGL-LifeSci model.
    
    Parameters:
        drugs_df: DataFrame containing 'drug_name' and 'smiles' columns.
        model_name: DGL-LifeSci pretrained model name.
        device: 'cuda' or 'cpu'. Automatically detected if None.
        
    Returns:
        pd.DataFrame: Contains ['drug_name', 'smiles', 'graph_0', ...]
    """
    _validate_drugs_df(drugs_df)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_drug = load_pretrained(model_name).to(device).eval()
    readout = AvgPooling()
    atom_featurizer = PretrainAtomFeaturizer()
    bond_featurizer = PretrainBondFeaturizer()

    N = len(drugs_df)
    graph_matrix = None
    bad_rows = []

    with torch.no_grad():
        for i, (drug_name, smi) in enumerate(
            tqdm(zip(drugs_df["drug_name"].astype(str), drugs_df["smiles"].astype(str)), total=N, desc="Graph Features")
        ):
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

            nfeats = [g.ndata.pop("atomic_number"), g.ndata.pop("chirality_type")]
            efeats = [g.edata.pop("bond_type"), g.edata.pop("bond_direction_type")]

            node_repr = model_drug(g, nfeats, efeats)
            pooled = readout(g, node_repr)
            vec = pooled.detach().cpu().numpy().reshape(-1).astype(np.float32)

            if graph_matrix is None:
                D = vec.shape[0]
                graph_matrix = np.zeros((N, D), dtype=np.float32)

            graph_matrix[i] = vec

    if bad_rows:
        raise ValueError(f"Invalid SMILES encountered: {bad_rows[:5]}")

    graph_cols = [f"graph_{j}" for j in range(graph_matrix.shape[1])]
    graph_df = pd.DataFrame(graph_matrix, columns=graph_cols)
    graph_df.insert(0, "smiles", drugs_df["smiles"].astype(str).values)
    graph_df.insert(0, "drug_name", drugs_df["drug_name"].astype(str).values)

    return graph_df

# ------------------------
# 4. Drug-Target Interaction (DTI) Features
# ------------------------

def _get_fasta_from_genes(target_genes: list, batch_size: int = 100) -> dict:
    """Helper function to fetch FASTA protein sequences from UniProt API."""
    fasta_dict = {}
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    for i in range(0, len(target_genes), batch_size):
        batch = target_genes[i : i + batch_size]
        query_parts = [f"gene_exact:{gene}" for gene in batch]
        query_string = "(" + " OR ".join(query_parts) + ") AND organism_id:9606 AND reviewed:true"

        params = {
            "query": query_string,
            "format": "json",
            "fields": "gene_names,sequence",
            "size": 500,
        }

        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        results = response.json().get("results", [])
        for entry in results:
            try:
                gene_symbol = entry["genes"][0]["geneName"]["value"]
                sequence = entry["sequence"]["value"]
                if gene_symbol in batch and gene_symbol not in fasta_dict:
                    fasta_dict[gene_symbol] = sequence
            except KeyError:
                continue

    return fasta_dict


def compute_dti_features(
    drugs_df: pd.DataFrame,
    raw_dir: Path,
    pretrained_model_path: Path,
    top_n_depmap: int = 500
) -> pd.DataFrame:
    """
    Computes Drug-Target Interaction affinity features using DeepPurpose.
    
    Parameters:
        drugs_df: DataFrame containing 'drug_name' and 'smiles' columns.
        raw_dir: Path object pointing to directory containing DepMap and LINCS files.
        pretrained_model_path: Path object pointing to the pretrained DeepPurpose model folder.
        top_n_depmap: Top N genes by variance to select from DepMap.
        
    Returns:
        pd.DataFrame: Matrix of shape (num_drugs, num_targets + 2) with drug_name and smiles.
    """
    from DeepPurpose import DTI as models
    from DeepPurpose import utils

    _validate_drugs_df(drugs_df)

    # Load target selection files
    crispr_gene_effect = pd.read_csv(raw_dir / 'CRISPRGeneEffect.csv', index_col=0)
    top_genes = crispr_gene_effect.var().sort_values(ascending=False).head(top_n_depmap)
    depmap_genes = [gene.split(" ")[0] for gene in top_genes.index]

    lincs = pd.read_csv(raw_dir / "GSE92742_Broad_LINCS_gene_info.txt", sep='\t')
    lincs_genes = lincs[lincs['pr_is_lm'] == 1]['pr_gene_symbol'].tolist()

    target_genes = sorted(set(depmap_genes + lincs_genes))

    # Query sequences
    target_dict = _get_fasta_from_genes(target_genes)

    # Construct drug-target pairs
    drug_names = drugs_df['drug_name'].tolist()
    drug_smiles = drugs_df['smiles'].tolist()
    target_names = sorted(target_dict)
    target_seqs = [target_dict[name] for name in target_names]

    X_drug, X_target = [], []
    pair_drug, pair_smiles, pair_target = [], [], []

    for d_name, d_smi in zip(drug_names, drug_smiles):
        for t_name, t_seq in zip(target_names, target_seqs):
            X_drug.append(d_smi)
            X_target.append(t_seq)
            pair_drug.append(d_name)
            pair_smiles.append(d_smi)
            pair_target.append(t_name)

    y_dummy = [0.0] * len(X_drug)

    # Load DeepPurpose Model
    mpnn_model = models.model_pretrained(path_dir=str(pretrained_model_path))

    # Process and Predict
    X_pred = utils.data_process(
        X_drug, X_target, y_dummy, 'MPNN', 'CNN', split_method='no_split'
    )
    pred_mpnn = mpnn_model.predict(X_pred)

    result = pd.DataFrame({
        'drug_name': pair_drug,
        'smiles': pair_smiles,
        'target_gene': pair_target,
        'pred_mpnn': pred_mpnn
    })

    # Pivot to drug x target matrix
    dti_matrix = result.pivot(index='drug_name', columns='target_gene', values='pred_mpnn')
    dti_df = dti_matrix.reset_index()

    dti_df = dti_df.merge(
        drugs_df[['drug_name', 'smiles']].drop_duplicates(),
        on='drug_name',
        how='left'
    )

    target_columns = [col for col in dti_df.columns if col not in ['drug_name', 'smiles']]
    dti_df = dti_df[['drug_name', 'smiles'] + target_columns]

    return dti_df

# ------------------------
# Compute drug features for training data and save
# ------------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DIR = PROJECT_ROOT / "raw"
    PROCESSED_DIR = PROJECT_ROOT / "processed"
    
    use_dti = False

    # Load data
    all_data = pd.read_csv(PROCESSED_DIR / 'Response_processed.csv')
    drugs = all_data[['drug_name', 'smiles']].drop_duplicates().reset_index(drop=True)

    # 1. Morgan Fingerprints
    fps_df = compute_morgan_fingerprints(drugs)
    fps_arr = fps_df.iloc[:, 2:].to_numpy(dtype=np.uint8)
    np.savez_compressed(
        PROCESSED_DIR / "drug_fingerprints.npz",
        fps=fps_arr,
        drug_name=drugs["drug_name"].astype(str).to_numpy(),
        smiles=drugs["smiles"].astype(str).to_numpy())

    # 2. ChemBERTa
    chem_df = compute_chemberta_embeddings(drugs)
    emb_arr = chem_df.iloc[:, 2:].to_numpy(dtype=np.float32)
    np.savez_compressed(
        PROCESSED_DIR / "drug_chemberta_embeddings.npz",
        emb=emb_arr,
        drug_name=drugs["drug_name"].astype(str).to_numpy(),
        smiles=drugs["smiles"].astype(str).to_numpy())

    # 3. Molecular Graphs
    graph_df = compute_graph_embeddings(drugs)
    graph_arr = graph_df.iloc[:, 2:].to_numpy(dtype=np.float32)
    np.savez_compressed(
        PROCESSED_DIR / "drug_molecular_graphs.npz",
        graph=graph_arr,
        drug_name=drugs["drug_name"].astype(str).to_numpy(),
        smiles=drugs["smiles"].astype(str).to_numpy())

    # 4. DTI Features
    if use_dti:
        model_path = PROJECT_ROOT / "save_folder" / "pretrained_models" / "mpnn_cnn_bindingdb_ic50"
        dti_df = compute_dti_features(drugs, RAW_DIR, model_path)
        dti_arr = dti_df.iloc[:, 2:].to_numpy(dtype=np.float32)
        np.savez_compressed(
            PROCESSED_DIR / "drug_dti.npz",
            dti=dti_arr,
            drug_name=dti_df["drug_name"].astype(str).to_numpy(),
            smiles=dti_df["smiles"].astype(str).to_numpy(),
            target_gene=np.asarray(dti_df.columns[2:], dtype=str))

    print("All features calculated and saved successfully.")