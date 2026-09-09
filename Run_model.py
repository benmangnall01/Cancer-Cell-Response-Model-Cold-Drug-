# ------------------------
# Import libraries
# ------------------------

import os
from pathlib import Path
import pandas as pd
from model import CDR_model
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split

# ------------------------
# Configuration
# ------------------------

# Choose setting: 'cold-drug' or 'cold-cell'
SCENARIO = 'cold-cell' 

if SCENARIO == 'cold-drug':
    SPLIT_COL = 'drug_name'
elif SCENARIO == 'cold-cell':
    SPLIT_COL = 'depmap_id'
else:
    raise ValueError("SCENARIO must be either 'cold-drug' or 'cold-cell'")

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUT_DIR = Path(f"cv_splits_{SCENARIO}") 
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Load data  
# ------------------------

# Load in master cell drug response table
pairs = pd.read_csv(PROCESSED_DIR / 'Response_processed.csv')
pairs = shuffle(pairs, random_state=2025) # shuffle data 

# Get a table of unique entities (either drugs or cells, depending on SCENARIO)
entities = pd.DataFrame(pairs[SPLIT_COL].unique(), columns=[SPLIT_COL])

# ------------------------
# 5-fold CV 
# ------------------------

n_splits = 5
val_frac_of_trainval = 0.125  # ~65/10/25 proportions

kf = KFold(n_splits=n_splits, shuffle=True, random_state=2025)

for fold, (trainval_idx, test_idx) in enumerate(kf.split(entities), start=1):
    entities_trainval = entities.iloc[trainval_idx].reset_index(drop=True)
    entities_test = entities.iloc[test_idx].reset_index(drop=True)

    # Split remaining drugs or cell lines into train and val
    entities_train, entities_val = train_test_split(
        entities_trainval,
        test_size=val_frac_of_trainval,
        random_state=2025 + fold,  # fold-specific but reproducible
        shuffle=True
    )

    # Split pairs by scenario
    pairs_train = pairs[pairs[SPLIT_COL].isin(entities_train[SPLIT_COL])]
    pairs_val   = pairs[pairs[SPLIT_COL].isin(entities_val[SPLIT_COL])]
    pairs_test  = pairs[pairs[SPLIT_COL].isin(entities_test[SPLIT_COL])]

    # Quick check
    print(f"Fold {fold}: {SCENARIO} train/val/test = {len(entities_train)}/{len(entities_val)}/{len(entities_test)}")

    # Save
    pairs_train.to_csv(OUT_DIR / f"train{fold}.csv", index=False)
    pairs_val.to_csv(OUT_DIR / f"val{fold}.csv", index=False)
    pairs_test.to_csv(OUT_DIR / f"test{fold}.csv", index=False)

print(f"Saved {n_splits} folds to: {OUT_DIR.resolve()}")

# ------------------------
# Run model for each fold 
# ------------------------ 

for i in range(1, n_splits + 1):
    print(f"Running fold {i} for {SCENARIO}...")
    
    # Load data
    train = pd.read_csv(OUT_DIR / f"train{i}.csv")
    val   = pd.read_csv(OUT_DIR / f"val{i}.csv")
    test  = pd.read_csv(OUT_DIR / f"test{i}.csv")

    # Run model 
    net = CDR_model()               
    net.train(train_drug=train, test_drug=test, val_drug=val)
