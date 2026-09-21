# ------------------------
# Import libraries
# ------------------------

from pathlib import Path
import pandas as pd
from model import CDR_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# ------------------------
# Configuration
# ------------------------

# SET SCENARIO HERE: 'cold-drug' or 'cold-cell'
SCENARIO = 'cold-drug' 

if SCENARIO == 'cold-drug':
    SPLIT_COL = 'smiles'
elif SCENARIO == 'cold-cell':
    SPLIT_COL = 'depmap_id'
else:
    raise ValueError("SCENARIO must be either 'cold-drug' or 'cold-cell'")

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUT_DIR / "Models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Load data  
# ------------------------

# Load in master cell drug response table
pairs = pd.read_csv(PROCESSED_DIR / 'Response_processed.csv')
pairs = shuffle(pairs, random_state=2026) # shuffle data 

# Get a table of unique entities (either drugs or cells, depending on SCENARIO)
entities = pd.DataFrame(pairs[SPLIT_COL].unique(), columns=[SPLIT_COL])

# ------------------------
# Single 90/5/5 split
# ------------------------

# 5% test, then split remaining 95% into 90% train and 5% val
entities_trainval, entities_test = train_test_split(
    entities,
    test_size=0.05,
    random_state=2025,
    shuffle=True,
)

val_frac_of_trainval = 0.05 / 0.95
entities_train, entities_val = train_test_split(
    entities_trainval,
    test_size=val_frac_of_trainval,
    random_state=2025,
    shuffle=True,
)

# Split pairs by entity membership (cold drug or cold cell)
pairs_train = pairs[pairs[SPLIT_COL].isin(entities_train[SPLIT_COL])]
pairs_val = pairs[pairs[SPLIT_COL].isin(entities_val[SPLIT_COL])]
pairs_test = pairs[pairs[SPLIT_COL].isin(entities_test[SPLIT_COL])]

print(f"Single split: {SCENARIO} train/val/test = {len(entities_train)}/{len(entities_val)}/{len(entities_test)}")
print(f"Counts by pair rows: {len(pairs_train)}/{len(pairs_val)}/{len(pairs_test)}")

# Save
pairs_train.to_csv(OUT_DIR / "train.csv", index=False)
pairs_val.to_csv(OUT_DIR / "val.csv", index=False)
pairs_test.to_csv(OUT_DIR / "test.csv", index=False)

print(f"Saved single split to: {OUT_DIR.resolve()}")

# ------------------------
# Train model once
# ------------------------

net = CDR_model()
net.train(train_drug=pairs_train, test_drug=pairs_test, val_drug=pairs_val)
net.save_model(MODEL_DIR / f'CDR_model_{SCENARIO}.pt')
