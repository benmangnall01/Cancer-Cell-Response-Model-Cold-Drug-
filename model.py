import os
import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from model_helper import Encoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"

# Global flags
use_crispr = True
use_dti = False

# -----------------------------
# Hard-coded defaults (locked)
# -----------------------------

_expr = pd.read_csv(PROCESSED_DIR / "expression.csv", index_col=0)
_mut  = pd.read_csv(PROCESSED_DIR / "mutations.csv", index_col=0)
_mrna = pd.read_csv(PROCESSED_DIR / "mrna.csv", index_col=0)
_cn   = pd.read_csv(PROCESSED_DIR / "copy_number.csv", index_col=0)
if use_crispr:
    _crispr = pd.read_csv(PROCESSED_DIR / "crispr.csv", index_col=0)

with np.load(PROCESSED_DIR / "drug_fingerprints.npz", allow_pickle=True) as f:
    fp_input_dim = f[f.files[0]].shape[1]
with np.load(PROCESSED_DIR / "drug_chemberta_embeddings.npz", allow_pickle=True) as f:
    seq_input_dim = f[f.files[0]].shape[1]
with np.load(PROCESSED_DIR / "drug_molecular_graphs.npz", allow_pickle=True) as f:
    graph_input_dim = f[f.files[0]].shape[1]
if use_dti:
    with np.load(PROCESSED_DIR / "drug_dti.npz", allow_pickle=True) as f:
        dti_input_dim = f[f.files[0]].shape[1]

DEFAULTS = {
    # dimensions
    "input_dim_expression": _expr.shape[1],
    "input_dim_mutation": _mut.shape[1],
    "input_dim_mrna": _mrna.shape[1],
    "input_dim_copy_number": _cn.shape[1],
    
    # cached drug features (precomputed)
    "fp_cache_path": PROCESSED_DIR / "drug_fingerprints.npz",
    "seq_cache_path": PROCESSED_DIR / "drug_chemberta_embeddings.npz",
    "graph_cache_path": PROCESSED_DIR / "drug_molecular_graphs.npz",

    # training defaults
    "lr": 1e-5,
    "decay": 0.0,
    "BATCH_SIZE": 256,
    "train_epoch": 100,
}

if use_crispr:
    DEFAULTS["input_dim_crispr"] = _crispr.shape[1]
if use_dti:
    DEFAULTS["dti_cache_path"] = PROCESSED_DIR / "drug_dti.npz"

# -----------------------------
# Cached drug feature loading
# -----------------------------
def load_cached_drug_features(fp_path: str, seq_path: str, graph_path: str, dti_path=None):
    """Load precomputed drug features from NPZ files and align them by SMILES."""
    def _npz(p: str):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cached drug feature file: {p}")
        return np.load(p, allow_pickle=True)

    fp_npz = _npz(fp_path)
    seq_npz = _npz(seq_path)
    g_npz  = _npz(graph_path)
    if use_dti:
        if dti_path is None:
            raise ValueError("A DTI cache path is required when use_dti=True.")
        dti_npz = _npz(dti_path)

    master_smiles = fp_npz["smiles"].astype(str)
    smiles_to_idx = {s: i for i, s in enumerate(master_smiles.tolist())}

    fps = fp_npz["fps"].astype(np.float32)  # saved as uint8; convert once
    if fps.ndim != 2 or len(fps) != len(master_smiles) or not np.isfinite(fps).all():
        raise ValueError("Fingerprint cache must contain one finite 2D row per SMILES.")

    # ---- Align ChemBERTa ----
    seq_smiles = seq_npz["smiles"].astype(str)
    seq = seq_npz["emb"].astype(np.float32)
    if seq.ndim != 2 or len(seq) != len(seq_smiles) or not np.isfinite(seq).all():
        raise ValueError("ChemBERTa cache must contain one finite 2D row per SMILES.")
    if (len(seq_smiles) != len(master_smiles)) or (not np.array_equal(seq_smiles, master_smiles)):
        seq_map = {s: i for i, s in enumerate(seq_smiles.tolist())}
        aligned = np.zeros((len(master_smiles), seq.shape[1]), dtype=np.float32)
        missing = 0
        for i, s in enumerate(master_smiles):
            j = seq_map.get(s)
            if j is None:
                missing += 1
            else:
                aligned[i] = seq[j]
        if missing:
            raise ValueError(f"ChemBERTa cache is missing {missing} fingerprint SMILES.")
        seq = aligned

    # ---- Align Graph embeddings ----
    g_smiles = g_npz["smiles"].astype(str)
    graph = g_npz["graph"].astype(np.float32)
    if graph.ndim != 2 or len(graph) != len(g_smiles) or not np.isfinite(graph).all():
        raise ValueError("Graph cache must contain one finite 2D row per SMILES.")
    if (len(g_smiles) != len(master_smiles)) or (not np.array_equal(g_smiles, master_smiles)):
        g_map = {s: i for i, s in enumerate(g_smiles.tolist())}
        aligned = np.zeros((len(master_smiles), graph.shape[1]), dtype=np.float32)
        missing = 0
        for i, s in enumerate(master_smiles):
            j = g_map.get(s)
            if j is None:
                missing += 1
            else:
                aligned[i] = graph[j]
        if missing:
            raise ValueError(f"Graph cache is missing {missing} fingerprint SMILES.")
        graph = aligned

    if use_dti:
        #---- Align DTI features ----
        dti_smiles = dti_npz["smiles"].astype(str)
        dti = dti_npz["dti"].astype(np.float32)

        if dti.ndim != 2 or len(dti) != len(dti_smiles) or not np.isfinite(dti).all():
            raise ValueError("DTI cache must contain one finite 2D row per SMILES.")

        if (len(dti_smiles) != len(master_smiles) or not np.array_equal(dti_smiles, master_smiles)):
            dti_map = {s: i for i, s in enumerate(dti_smiles.tolist())}
            aligned = np.zeros((len(master_smiles), dti.shape[1]), dtype=np.float32)
            missing = 0

            for i, s in enumerate(master_smiles):
                j = dti_map.get(s)
                if j is None:
                    missing += 1
                else:
                    aligned[i] = dti[j]

            if missing:
                raise ValueError(f"DTI cache is missing {missing} fingerprint SMILES.")
            dti = aligned

    return {"fps": fps, "seq": seq, "graph": graph, **({"dti": dti} if use_dti else {}), "smiles_to_idx": smiles_to_idx}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset loader (defaults-only)
# -----------------------------

class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df, cached_drug_features=None, expression_data=None):
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df = drug_df

        # Cell
        self.expression_data = _expr if expression_data is None else expression_data
        self.mutation_data   = _mut
        self.mrna_data= _mrna
        self.copy_number_data= _cn
        if use_crispr:
            self.crispr_data = _crispr

        # Cached drug features
        self.cached_drug_features = cached_drug_features
        self.smiles_to_idx = cached_drug_features["smiles_to_idx"]
        self.drug_fp   = cached_drug_features["fps"]
        self.drug_seq  = cached_drug_features["seq"]
        self.drug_graph= cached_drug_features["graph"]
        if use_dti:
            self.drug_dti = cached_drug_features["dti"]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        index = self.list_IDs[idx]
        y = self.labels[idx]

        depmap_id = str(self.drug_df.iloc[index]["depmap_id"]).strip()

        v_expression = np.array(self.expression_data.loc[depmap_id, :], dtype=np.float32)
        v_mutation   = np.array(self.mutation_data.loc[depmap_id, :], dtype=np.float32)
        v_mrna= np.array(self.mrna_data.loc[depmap_id, :], dtype=np.float32)
        v_copy_number= np.array(self.copy_number_data.loc[depmap_id, :], dtype=np.float32)
        if use_crispr:
            v_crispr = np.array(self.crispr_data.loc[depmap_id, :], dtype=np.float32)

        smiles = str(self.drug_df.iloc[index]["smiles"])
        j = self.smiles_to_idx.get(smiles)
        if j is None:
            raise KeyError(f"No cached drug features found for SMILES: {smiles}")
            
        v_fp    = self.drug_fp[j]
        v_seq   = self.drug_seq[j]
        v_graph = self.drug_graph[j]
        if use_dti:
            v_dti = self.drug_dti[j]

        return (
            torch.from_numpy(v_fp),
            torch.from_numpy(v_seq),
            torch.from_numpy(v_graph),
            *( (torch.from_numpy(v_dti),) if use_dti else () ),
            torch.from_numpy(v_expression),
            torch.from_numpy(v_mutation),
            torch.from_numpy(v_mrna),
            torch.from_numpy(v_copy_number),
            *( (torch.from_numpy(v_crispr),) if use_crispr else () ),
            torch.tensor(y, dtype=torch.float32),
        )

# -----------------------------
# Model blocks
# -----------------------------

class MLP(nn.Sequential):
    def __init__(self, input_dim_gene: int):
        super().__init__()
        hidden_dim_gene = 256
        mlp_hidden_dims_gene = [1024, 512]
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        v = v.float().to(device)
        for l in self.predictor:
            v = F.relu(l(v))
        return v

class Classifier(nn.Sequential):
    def __init__(self):
        super().__init__()
        
        # Save flags as class attributes
        self.use_dti = use_dti
        self.use_crispr = use_crispr

        # Drug pretrained projections
        self.model_fp = MLP(fp_input_dim)
        self.model_seq = MLP(seq_input_dim)
        self.model_graph = MLP(graph_input_dim)
        if self.use_dti:
            self.model_dti = MLP(dti_input_dim)

        # Cell projections
        self.model_expression = MLP(DEFAULTS["input_dim_expression"])
        self.model_mutation = MLP(DEFAULTS["input_dim_mutation"])
        self.model_mrna = MLP(DEFAULTS["input_dim_mrna"])
        self.model_copy_number = MLP(DEFAULTS["input_dim_copy_number"])
        if self.use_crispr:
            self.model_crispr = MLP(DEFAULTS["input_dim_crispr"])

        # Determine dynamic number of modalities/tokens
        num_tokens = 7
        if self.use_dti: num_tokens += 1
        if self.use_crispr: num_tokens += 1

        # Encoder fusion (dynamic sequence length)
        self.fusion = Encoder(256, 256, 8, 6, 0.1, device)

        # Head
        hidden_dims = [1024, 1024, 512]
        dims = [256 * num_tokens] + hidden_dims + [1]
        
        self.dropout = nn.Dropout(0.1)
        layer_size = len(hidden_dims) + 1
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        label = v[-1]
        v_iter = iter(v)

        # ---- Drug ----
        v_fp = self.model_fp(next(v_iter).to(device)).unsqueeze(1)
        v_seq = self.model_seq(next(v_iter).to(device)).unsqueeze(1)
        v_graph = self.model_graph(next(v_iter).to(device)).unsqueeze(1)
        
        drug_tensors = [v_fp, v_seq, v_graph]
        
        if self.use_dti:
            v_dti = self.model_dti(next(v_iter).to(device)).unsqueeze(1)
            drug_tensors.append(v_dti)
            
        v_D = torch.cat(drug_tensors, dim=1)

        # ---- Cell ----
        v_expression = self.model_expression(next(v_iter)).unsqueeze(1)
        v_mutation = self.model_mutation(next(v_iter)).unsqueeze(1)
        v_mrna = self.model_mrna(next(v_iter)).unsqueeze(1)
        v_copy_number = self.model_copy_number(next(v_iter)).unsqueeze(1)
        
        cell_tensors = [v_expression, v_mutation, v_mrna, v_copy_number]
        
        if self.use_crispr:
            v_crispr = self.model_crispr(next(v_iter)).unsqueeze(1)
            cell_tensors.append(v_crispr)
            
        v_cell = torch.cat(cell_tensors, dim=1)

        # ---- Fusion (encoder only) ----
        v_f = torch.cat((v_D, v_cell), 1)  # (B, num_tokens, 256)
        v_f = self.fusion(v_f, None)
        v_f = v_f.view(-1, v_f.shape[1] * v_f.shape[2])  # (B, num_tokens*256)

        # ---- Head ----
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))

        return v_f, label

# -----------------------------
# Trainer / Wrapper
# -----------------------------

class CDR_model:
    def __init__(self, **_ignored_config):
        # ignore passed config on purpose
        self.config = dict(DEFAULTS)
        self.model = Classifier().to(device)
        self.device = device
        self.expression_scaler = None
        self.expression_data = _expr

    def test(self, datagenerator, model):
        y_label = []
        y_pred = []
        was_training = model.training
        model.eval()

        with torch.no_grad():
            for _, v in enumerate(datagenerator):
                score, label = model(v)
                logits = torch.squeeze(score).detach().cpu().numpy()
                label_ids = label.to("cpu").numpy()
                y_label += label_ids.flatten().tolist()
                y_pred += logits.flatten().tolist()

        if was_training:
            model.train()

        mse = mean_squared_error(y_label, y_pred)

        return (
            y_label,
            y_pred,
            mse,
            np.sqrt(mse),
            pearsonr(y_label, y_pred)[0],
            pearsonr(y_label, y_pred)[1],
            spearmanr(y_label, y_pred)[0],
            spearmanr(y_label, y_pred)[1],
            concordance_index(y_label, y_pred),
        )

    def save_model(self, path="saved_model.pt"):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        saved_config = {key: value.name if isinstance(value, Path) else value for key, value in self.config.items()}
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": saved_config,
            "use_crispr": use_crispr,
            "use_dti": use_dti,
            "feature_columns": {
                "expression": _expr.columns.tolist(),
                "mutation": _mut.columns.tolist(),
                "mrna": _mrna.columns.tolist(),
                "copy_number": _cn.columns.tolist(),
                **({"crispr": _crispr.columns.tolist()} if use_crispr else {}),
            },
            "drug_feature_dimensions": {
                "fingerprints": fp_input_dim,
                "chemberta": seq_input_dim,
                "graphs": graph_input_dim,
                **({"dti": dti_input_dim} if use_dti else {}),
            },
            "expression_scaler": self.expression_scaler,
        }
        torch.save(checkpoint, save_path)
    
    def train(self, train_drug, test_drug=None, val_drug=None):
        required_columns = {"depmap_id", "smiles", "lnIC50"}
        for name, frame in (("train", train_drug), ("test", test_drug), ("validation", val_drug)):
            if frame is None:
                continue
            missing_columns = required_columns - set(frame.columns)
            if missing_columns:
                raise ValueError(f"{name} data is missing columns: {sorted(missing_columns)}")
            if frame.empty:
                raise ValueError(f"{name} data must not be empty.")
            if frame[["depmap_id", "smiles", "lnIC50"]].isna().any().any() or not np.isfinite(pd.to_numeric(frame["lnIC50"], errors="coerce")).all():
                raise ValueError(f"{name} data contains missing or non-finite required values.")

        lr = self.config["lr"]
        decay = self.config["decay"]
        BATCH_SIZE = self.config["BATCH_SIZE"]
        train_epoch = self.config["train_epoch"]
 
        self.model = self.model.to(self.device)
 
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=decay,
        )
        loss_history = []
 
        # ---- Load cached drug features once (avoid recomputing per dataset/fold) ----
        fp_path = self.config.get("fp_cache_path", "drug_fingerprints.npz")
        seq_path = self.config.get("seq_cache_path", "drug_chemberta_embeddings.npz")
        graph_path = self.config.get("graph_cache_path", "drug_molecular_graphs.npz")
        dti_path = self.config.get("dti_cache_path")
        cached_drug_features = load_cached_drug_features(fp_path, seq_path, graph_path, dti_path)

        train_cell_ids = train_drug["depmap_id"].astype(str).unique()
        missing_cells = sorted(set(train_cell_ids) - set(_expr.index.astype(str)))
        if missing_cells:
            raise KeyError(f"Training data contains cell lines without expression features: {missing_cells[:5]}")

        scaler = StandardScaler().fit(_expr.loc[train_cell_ids])
        self.expression_scaler = {"mean": scaler.mean_, "scale": scaler.scale_}
        self.expression_data = pd.DataFrame(scaler.transform(_expr), index=_expr.index, columns=_expr.columns)
 
        train_ds = data_process_loader(list_IDs=np.arange(len(train_drug)), labels=train_drug["lnIC50"].values, drug_df=train_drug.reset_index(drop=True), cached_drug_features=cached_drug_features, expression_data=self.expression_data)

        training_generator = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, pin_memory=(device.type == "cuda"))
 
        testing_generator = None
        validation_generator = None
 
        if test_drug is not None:
            test_ds = data_process_loader(list_IDs=np.arange(len(test_drug)), labels=test_drug["lnIC50"].values, drug_df=test_drug.reset_index(drop=True), cached_drug_features=cached_drug_features, expression_data=self.expression_data)
            testing_generator = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, pin_memory=(device.type == "cuda"))
 
        if val_drug is not None:
            val_ds = data_process_loader(list_IDs=np.arange(len(val_drug)), labels=val_drug["lnIC50"].values, drug_df=val_drug.reset_index(drop=True), cached_drug_features=cached_drug_features, expression_data=self.expression_data)
            validation_generator = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, pin_memory=(device.type == "cuda"))

        # regression-only validation selection
        max_mse = float("inf")
 
        model_max = copy.deepcopy(self.model) if validation_generator is not None else None
        float2str = lambda x: "%0.4f" % x
 
        # Training
        t_start = time.time()
        iteration_loss = 0
 
        es = 0
        for epo in range(train_epoch):
            for i, v in enumerate(training_generator):
                score, label = self.model(v)
                label = label.float().to(self.device).view(-1)
 
                n = score.squeeze(1)                            
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(n, label)
 
                loss_history.append(loss.item())
                iteration_loss += 1
 
                opt.zero_grad()
                loss.backward()
                opt.step()
 
            if validation_generator is not None:
                with torch.set_grad_enabled(False):
                    y_true, y_pred, mse, rmse, pearson, p_val, spearman, s_p_val, CI = self.test(validation_generator, self.model)
 
                    lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, pearson, p_val, spearman, s_p_val, CI]))
                    t_now = time.time()
 
                    if mse < max_mse:
                        model_max = copy.deepcopy(self.model)
                        max_mse = mse
                        es = 0
                        # Display evaluation metrics
                        print("Validation at Epoch " + str(epo + 1) + " with MSE: " + str(mse)[:7]
                            + ", Pearson Correlation: " + str(pearson)[:7] + " and Spearman Correlation: " + str(spearman)[:7]
                            + ", Total time " + str(int(t_now - t_start) / 60)[:7] + " minutes")
                    else:
                        es += 1
                        # Display evaluation metrics
                        print("Validation at Epoch " + str(epo + 1) + " with MSE: " + str(mse)[:7]
                            + ", Pearson Correlation: " + str(pearson)[:7] + " and Spearman Correlation: " + str(spearman)[:7]
                            + ", Total time " + str(int(t_now - t_start) / 60)[:7] + " minutes" + f", Counter {es} of 5")
                        if es > 4:
                            print("Early stopping with best MSE: " + str(max_mse)[:7] + " and MSE for this epoch: " + str(mse)[:7] + " ...")
                            break
 
        # Load the best validation model, or retain the final trained weights.
        if model_max is not None:
            self.model = model_max
        else:
            model_max = self.model
 
        # ------------------------
        # Testing
        # ------------------------
 
        if testing_generator is not None:
            y_true, y_pred, mse, rmse, pearson, p_val, spearman, s_p_val, CI = self.test(testing_generator, model_max)
            print("Testing MSE: " + str(mse) + " , Pearson Correlation: " + str(pearson) + " , Spearman Correlation: " + str(spearman) +  " , Concordance Index: " + str(CI) )

            return {"mse": mse, "rmse": rmse, "pearson": pearson, "spearman": spearman, "concordance_index": CI}

        return None
