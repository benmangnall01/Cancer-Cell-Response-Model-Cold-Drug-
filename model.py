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
_meth = pd.read_csv(PROCESSED_DIR / "methylation.csv", index_col=0)
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
    "input_dim_methylation": _meth.shape[1],
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
def load_cached_drug_features(fp_path: str, seq_path: str, graph_path: str, dti_path: str | None = None):
    """Load precomputed drug features from NPZ files and align them by SMILES."""
    def _npz(p: str):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cached drug feature file: {p}")
        return np.load(p, allow_pickle=True)

    fp_npz = _npz(fp_path)
    seq_npz = _npz(seq_path)
    g_npz  = _npz(graph_path)
    if use_dti:
        dti_npz = _npz(dti_path)

    master_smiles = fp_npz["smiles"].astype(str)
    smiles_to_idx = {s: i for i, s in enumerate(master_smiles.tolist())}

    fps = fp_npz["fps"].astype(np.float32)  # saved as uint8; convert once

    # ---- Align ChemBERTa ----
    seq_smiles = seq_npz["smiles"].astype(str)
    seq = seq_npz["emb"].astype(np.float32)
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
            print(f"[WARN] ChemBERTa cache missing {missing} SMILES (filled with zeros).")
        seq = aligned

    # ---- Align Graph embeddings ----
    g_smiles = g_npz["smiles"].astype(str)
    graph = g_npz["graph"].astype(np.float32)
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
            print(f"[WARN] Graph cache missing {missing} SMILES (filled with zeros).")
        graph = aligned

    if use_dti:
        #---- Align DTI features ----
        dti_smiles = dti_npz["smiles"].astype(str)
        dti = dti_npz["dti"].astype(np.float32)

        if dti.ndim != 2:
            raise ValueError(f"Expected DTI array to be 2D, got shape {dti.shape}")

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
                print(f"[WARN] DTI cache missing {missing} SMILES (filled with zeros).")
            dti = aligned

    return {"fps": fps, "seq": seq, "graph": graph, **({"dti": dti} if use_dti else {}), "smiles_to_idx": smiles_to_idx}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset loader (defaults-only)
# -----------------------------

class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df, cached_drug_features=None):
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df = drug_df

        # Cell
        self.expression_data = _expr
        self.mutation_data   = _mut
        self.methylation_data= _meth
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
        v_methylation= np.array(self.methylation_data.loc[depmap_id, :], dtype=np.float32)
        v_copy_number= np.array(self.copy_number_data.loc[depmap_id, :], dtype=np.float32)
        if use_crispr:
            v_crispr = np.array(self.crispr_data.loc[depmap_id, :], dtype=np.float32)

        smiles = str(self.drug_df.iloc[index]["smiles"])
        j = self.smiles_to_idx.get(smiles)
            
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
            torch.from_numpy(v_methylation),
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
        self.model_methylation = MLP(DEFAULTS["input_dim_methylation"])
        self.model_copy_number = MLP(DEFAULTS["input_dim_copy_number"])
        if self.use_crispr:
            self.model_crispr = MLP(DEFAULTS["input_dim_crispr"])

        # Determine dynamic number of modalities/tokens
        num_tokens = 7
        if self.use_dti: num_tokens += 1
        if self.use_crispr: num_tokens += 1

        # Encoder fusion (dynamic sequence length)
        self.fusion = Encoder(256, 256, num_tokens, 6, 0.1, device)

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
        v_methylation = self.model_methylation(next(v_iter)).unsqueeze(1)
        v_copy_number = self.model_copy_number(next(v_iter)).unsqueeze(1)
        
        cell_tensors = [v_expression, v_mutation, v_methylation, v_copy_number]
        
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

    def test(self, datagenerator, model):
        y_label = []
        y_pred = []
        model.eval()

        for _, v in enumerate(datagenerator):
            score, label = model(v)

            # regression only
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            label_t = label.float().to(self.device).view(-1)
            loss = loss_fct(n, label_t)

            logits = torch.squeeze(score).detach().cpu().numpy()

            label_ids = label.to("cpu").numpy()
            y_label += label_ids.flatten().tolist()
            y_pred += logits.flatten().tolist()

        model.train()

        return (
            y_label,
            y_pred,
            mean_squared_error(y_label, y_pred),
            np.sqrt(mean_squared_error(y_label, y_pred)),
            pearsonr(y_label, y_pred)[0],
            pearsonr(y_label, y_pred)[1],
            spearmanr(y_label, y_pred)[0],
            spearmanr(y_label, y_pred)[1],
            concordance_index(y_label, y_pred),
            loss,
        )
    
    def train(self, train_drug, test_drug, val_drug):
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
        cached_drug_features = None
        fp_path = self.config.get("fp_cache_path", "drug_fingerprints.npz")
        seq_path = self.config.get("seq_cache_path", "drug_chemberta_embeddings.npz")
        graph_path = self.config.get("graph_cache_path", "drug_molecular_graphs.npz")
        cached_drug_features = load_cached_drug_features(fp_path, seq_path, graph_path)
    
        params = {
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "num_workers": 0,
            "drop_last": False,
        }
 
        train_ds = data_process_loader(list_IDs=np.arange(len(train_drug)), labels=train_drug["lnIC50"].values, drug_df=train_drug.reset_index(drop=True), cached_drug_features=cached_drug_features)

        set(cached_drug_features["smiles_to_idx"].keys())

        training_generator = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, pin_memory=(device.type == "cuda"))
 
        testing_generator = None
        validation_generator = None
 
        if test_drug is not None:
            test_ds = data_process_loader(list_IDs=np.arange(len(test_drug)), labels=test_drug["lnIC50"].values, drug_df=test_drug.reset_index(drop=True), cached_drug_features=cached_drug_features)
            testing_generator = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, pin_memory=(device.type == "cuda"))
 
        if val_drug is not None:
            val_ds = data_process_loader(list_IDs=np.arange(len(val_drug)), labels=val_drug["lnIC50"].values, drug_df=val_drug.reset_index(drop=True), cached_drug_features=cached_drug_features)
            validation_generator = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, pin_memory=(device.type == "cuda"))

        # regression-only validation selection
        max_mse = 1000000
 
        model_max = copy.deepcopy(self.model)
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
                    y_true, y_pred, mse, rmse, pearson, p_val, spearman, s_p_val, CI, loss_val = self.test(validation_generator, self.model)
 
                    lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, pearson, p_val, spearman, s_p_val, CI]))
                    t_now = time.time()
 
                    if mse < max_mse:
                        model_max = copy.deepcopy(self.model)
                        max_mse = mse
                        es = 0
                        # Display evaluation metrics
                        print("Validation at Epoch " + str(epo + 1) + " with loss:" + str(loss_val.item())[:7] + ", MSE: " + str(mse)[:7]
                            + ", Pearson Correlation: " + str(pearson)[:7] + " Spearman Correlation: " + str(CI)[:7]
                            + ", Total time " + str(int(t_now - t_start) / 60)[:7] + " minutes")
                    else:
                        es += 1
                        # Display evaluation metrics
                        print("Validation at Epoch " + str(epo + 1) + " with loss:" + str(loss_val.item())[:7] + ", MSE: " + str(mse)[:7]
                            + ", Pearson Correlation: " + str(pearson)[:7] + " Spearman Correlation: " + str(CI)[:7]
                            + ", Total time " + str(int(t_now - t_start) / 60)[:7] + " minutes" + f", Counter {es} of 5")
                        if es > 4:
                            print("Early stopping with best MSE: " + str(max_mse)[:7] + " and MSE for this epoch: " + str(mse)[:7] + " ...")
                            break
 
        # load best model
        self.model = model_max
 
        # ------------------------
        # Testing
        # ------------------------
 
        if testing_generator is not None:
            y_true, y_pred, mse, rmse, pearson, p_val, spearman, s_p_val, CI, loss_test = self.test(testing_generator, model_max)
            #test_table = PrettyTable(["MSE", "RMSE", "Pearson Correlation", "p-value", "spearman", "s_p-value", "Concordance Index"])
            #test_table.add_row(list(map(float2str, [mse, rmse, pearson, p_val, spearman, s_p_val, CI])))
 
            print("Testing MSE: " + str(mse) + " , Pearson Correlation: " + str(pearson) + " , Spearman Correlation: " + str(spearman) +  " , Concordance Index: " + str(CI) )
