# ------------------------
# Import libraries
# ------------------------

import os
from pathlib import Path
import pandas as pd
from model import TransCDR
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUT_DIR = Path("cv_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Load data  
# ------------------------

# Load in master cell drug response table
pairs = pd.read_csv(PROCESSED_DIR / 'Response_processed.csv')
pairs = shuffle(pairs,random_state=2025) # shuffle data 

# Get a table of unique drugs
drugs = pd.DataFrame(pairs['drug_name'].unique()).rename(columns={0:'drug_name'})

# ------------------------
# 5-fold CV 
# ------------------------

n_splits = 5
val_frac_of_trainval = 0.125  # ~65/10/25-ish proportions

kf = KFold(n_splits=n_splits, shuffle=True, random_state=2025)

for fold, (trainval_idx, test_idx) in enumerate(kf.split(drugs), start=1):
    drugs_trainval = drugs.iloc[trainval_idx].reset_index(drop=True)
    drugs_test = drugs.iloc[test_idx].reset_index(drop=True)

    # Split remaining drugs into train and val
    drugs_train, drugs_val = train_test_split(
        drugs_trainval,
        test_size=val_frac_of_trainval,
        random_state=2025 + fold,  # fold-specific but reproducible
        shuffle=True)

    # Split pairs by drug membership (cold drug)
    pairs_train = pairs[pairs["drug_name"].isin(drugs_train["drug_name"])]
    pairs_val   = pairs[pairs["drug_name"].isin(drugs_val["drug_name"])]
    pairs_test  = pairs[pairs["drug_name"].isin(drugs_test["drug_name"])]

    # Quick check
    print(f"Fold {fold}: drugs train/val/test = "f"{len(drugs_train)}/{len(drugs_val)}/{len(drugs_test)}")

    # Save
    pairs_train.to_csv(OUT_DIR / f"train{fold}.csv", index=False)
    pairs_val.to_csv(OUT_DIR / f"val{fold}.csv", index=False)
    pairs_test.to_csv(OUT_DIR / f"test{fold}.csv", index=False)

print(f"Saved {n_splits} folds to: {OUT_DIR.resolve()}")

# ------------------------
# Run model for each fold 
# ------------------------ 

for i in range(1, n_splits + 1):
    print(f"Running fold {i}...")
    
    # Load data
    train = pd.read_csv(OUT_DIR / f"train{i}.csv")
    val   = pd.read_csv(OUT_DIR / f"val{i}.csv")
    test  = pd.read_csv(OUT_DIR / f"test{i}.csv")

    # Run model 
    net = CDR_model()               
    net.train(train_drug=train, test_drug=test, val_drug=val)
