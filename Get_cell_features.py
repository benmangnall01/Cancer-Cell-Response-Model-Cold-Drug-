# ------------------------
# Load packages
# ------------------------

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"

# Choose to use CRISPR data as a feature, or prism data for the training data set
use_crispr = True
use_prism = False

# ------------------------
# Load GDSC
# ------------------------

gdsc = pd.read_csv(RAW_DIR / 'GDSC2_fitted_dose_response_27Oct23.csv') 
gdsc = gdsc[['COSMIC_ID', 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']]
gdsc.columns = ['COSMIC_ID', 'cell_type', 'drug_name', 'lnIC50']

# ------------------------
# Load their data set
# ------------------------

transcdr = pd.read_csv(RAW_DIR / 'CDR_n156813.txt',sep='\t',index_col=0)
transcdr_col = transcdr[['COSMIC_ID', 'assay_name', 'GDSC_tissue']].drop_duplicates()
transcdr = transcdr.drop(columns = ['drug_id', 'cancer_type', 'smiles'])
gdsc = gdsc.merge(transcdr_col, on = 'COSMIC_ID', how = 'left')
gdsc = pd.concat([gdsc, transcdr]).dropna().drop_duplicates(subset = ['COSMIC_ID', 'drug_name'])

# ------------------------
# Join SMILES to main df 
# ------------------------

smiles = pd.read_csv(PROCESSED_DIR / 'drug_smiles.csv') 
smiles = smiles.rename(columns={"drug": "drug_name"})

def normalize_name(x: str) -> str:
    return (
        str(x)
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
    )

gdsc["drug_name"] = gdsc["drug_name"].map(normalize_name)

gdsc = gdsc.merge(smiles, on='drug_name', how='left')

# ------------------------
# Nomalise cell lines names 
# ------------------------

model = pd.read_csv(RAW_DIR / "model.csv")

model = model[["ModelID", "CellLineName", "StrippedCellLineName"]].dropna()

model = model.rename(columns={
    "ModelID": "depmap_id",
    "CellLineName": "cell_line_name",
    "StrippedCellLineName": "stripped_cell_line_name",
})

model["norm_ccle"] = model["cell_line_name"].map(normalize_name)
model["norm_stripped"] = model["stripped_cell_line_name"].map(normalize_name)
gdsc["norm_cell"] = gdsc["cell_type"].map(normalize_name)

# Identify norm_cell values that map to multiple original cell_types
duplicate_norm_cells = gdsc.groupby('norm_cell')['cell_type'].nunique()
duplicate_norm_cells = duplicate_norm_cells[duplicate_norm_cells > 1].index

# Remove rows where norm_cell has duplicates
gdsc = gdsc[~gdsc['norm_cell'].isin(duplicate_norm_cells)]

# Create a unified lookup by stacking both normalized name columns
model_lookup = pd.concat([
    model[['norm_ccle', 'depmap_id']].rename(columns={'norm_ccle': 'norm_name'}),
    model[['norm_stripped', 'depmap_id']].rename(columns={'norm_stripped': 'norm_name'})
]).drop_duplicates(subset=['norm_name'], keep='first')

# Now merge with gdsc
gdsc = gdsc.merge(
    model_lookup,
    left_on='norm_cell',
    right_on='norm_name',
    how='left').drop(columns=['norm_name', 'norm_cell'])

# Identify depmap_ids that have more than 1 unique assay_name
duplicate_depmap_ids = gdsc.groupby('depmap_id')['assay_name'].nunique()
duplicate_depmap_ids = duplicate_depmap_ids[duplicate_depmap_ids > 1].index

# Remove rows where depmap_id has multiple assay_names
gdsc = gdsc[~gdsc['depmap_id'].isin(duplicate_depmap_ids)]

# ------------------------
# Load cell features
# ------------------------

# Get lists of depmap_id that are in the data 
cell_list = gdsc['depmap_id'].unique().tolist()

# Load data sets
expression_raw = pd.read_csv(RAW_DIR / 'CCLE_expression.csv')
mutations_raw = pd.read_csv(RAW_DIR / 'CCLE_mutations.csv', sep="\t")
mrna_raw = pd.read_csv(RAW_DIR / 'mrna_n20617_1028_zscore.csv',index_col=0)
cn_raw = pd.read_csv(RAW_DIR / 'CCLE_gene_cn.csv')
if use_crispr:
    crispr_raw = pd.read_csv(RAW_DIR / 'Achilles_gene_effect.csv')

print('Loaded raw data sets')

# ------------------------
# Some expression data cleaing
# ------------------------

expression = expression_raw.rename(columns={'Unnamed: 0': "depmap_id"})
expression.columns = expression.columns.str.split(' ').str[0]
columns_to_scale = expression.columns.drop('depmap_id')
scaler = StandardScaler()
expression[columns_to_scale] = scaler.fit_transform(expression[columns_to_scale])
expression = expression[expression['depmap_id'].isin(cell_list)]

print('Got expression features')

# ------------------------
# Some mutation data cleaning 
# ------------------------   

# Define mutations 
nonsilent_classes = [
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "Missense_Mutation", "Nonsense_Mutation",
    "Nonstop_Mutation", "Splice_Site",
    "Translation_Start_Site",
    "In_Frame_Del", "In_Frame_Ins"]

# Build mutaions data frame
mut = mutations_raw[mutations_raw['Variant_Classification'].isin(nonsilent_classes)]
mut = mut[['DepMap_ID', 'Hugo_Symbol']].rename(columns = {'DepMap_ID': 'depmap_id'})
mut["mut_flag"] = 1

# Pivot to wide format where cell lines = rows and genes = cols
mut_pivot = mut.pivot_table(
    index='depmap_id',
    columns='Hugo_Symbol',
    values="mut_flag",
    aggfunc="max",
    fill_value=0,
)

# Remove the top level of the column MultiIndex
mut_pivot.columns.name = None
mut_pivot = mut_pivot.reset_index()

# Include only cell lines that we have data for
mutations = mut_pivot[mut_pivot['depmap_id'].isin(cell_list)]

print('Got mutation features')

# ------------------------
# Some methylation data cleaning 
# ------------------------

# Get list of cell lines with mRNA data
cell_list_mrna = gdsc['cell_type'].unique().tolist()

# Convert index to depmap_id
mrna = mrna_raw.T.reset_index()
mrna = mrna.rename(columns={'index': 'cell_type'})
mrna = mrna[mrna['cell_type'].isin(cell_list_mrna)]
mrna_key = gdsc[['depmap_id', 'cell_type']].drop_duplicates()
mrna = mrna.merge(mrna_key, on = 'cell_type', how = 'left').drop(columns = 'cell_type').dropna()
mrna.insert(0, 'depmap_id', mrna.pop('depmap_id'))

print('Got methylation features')

# ------------------------
# Some copy number data cleaning 
# ------------------------

cn = cn_raw.rename(columns={'Unnamed: 0': "depmap_id"})
cn = cn[cn['depmap_id'].isin(cell_list)]

print('Got copy number features')

# ------------------------
# Some crispr data cleaning 
# ------------------------

if use_crispr:
    crispr = crispr_raw.rename(columns={'DepMap_ID': "depmap_id"})
    crispr = crispr[crispr['depmap_id'].isin(cell_list)]

    # Remove na's
    def optimize_dropna_greedy(df, iterations=10000):
        """
        Greedy algorithm: each iteration, drop the single column OR row with the most NAs
        This ensures we always shrink and maximize data retention
        """
        df_clean = df.copy()
        
        for i in range(iterations):
            initial_shape = df_clean.shape
            remaining_nas = df_clean.isna().sum().sum()
            
            if remaining_nas == 0:
                print(f"No NAs remaining after {i} iterations!")
                break
                
            # Calculate NA counts for columns and rows
            col_na_counts = df_clean.isna().sum(axis=0)
            row_na_counts = df_clean.isna().sum(axis=1)
            
            max_col_nas = col_na_counts.max()
            max_row_nas = row_na_counts.max()

            # Apply weight to row NAs to control preference
            max_row_nas = max_row_nas * 0.01 #if set to 0.001 no cell lines get cut (16184 columns), if 0.01 then 4 get cut (17414 columns).
                                                    
            # Decide whether to drop a column or row (whichever removes more NAs)
            if max_col_nas >= max_row_nas:
                # Drop the column with most NAs
                col_to_drop = col_na_counts.idxmax()
                df_clean = df_clean.drop(columns=[col_to_drop])
                
            else:
                # Drop the row with most NAs
                row_to_drop = row_na_counts.idxmax()
                df_clean = df_clean.drop(index=[row_to_drop])
                
            if i % 10 == 0 or i < 10:
                remaining_nas = df_clean.isna().sum().sum()
        
        return df_clean

    # Run the greedy algorithm
    crispr = optimize_dropna_greedy(crispr, iterations=10000) # should be 710 iterations if 0.01
    #crispr = crispr.dropna(axis=1) # to get all the cell lines (16184 columns)

    print('Got CRISPR features')

# ------------------------
# Filter to cells in all 3 omics + smiles present
# ------------------------ 

# Filter GDSC to rows with complete data
gdsc_filtered = gdsc.dropna(subset=["depmap_id", "smiles", "lnIC50"]).drop(columns = 'cell_type').copy()

# Get sets of cells in each data set
cells_expr = set(expression["depmap_id"].unique())
cells_meth = set(mrna["depmap_id"].unique())
cells_mut  = set(mutations["depmap_id"].unique())
cells_cn   = set(cn["depmap_id"].unique())
if use_crispr:
    cells_crispr = set(crispr["depmap_id"].unique())

# Filter each data set to only common cells 
if use_crispr:
    common_cells = cells_expr & cells_meth & cells_mut & cells_cn & cells_crispr
else:
    common_cells = cells_expr & cells_meth & cells_mut & cells_cn 
gdsc_filtered = gdsc_filtered[gdsc_filtered["depmap_id"].isin(common_cells)]
expression = expression[expression["depmap_id"].isin(common_cells)]
mrna = mrna[mrna["depmap_id"].isin(common_cells)]
mutations = mutations[mutations["depmap_id"].isin(common_cells)]
cn = cn[cn["depmap_id"].isin(common_cells)]
if use_crispr:
    crispr = crispr[crispr['depmap_id'].isin(common_cells)]

# Dedupe by (cell, drug) 
gdsc_filtered = (gdsc_filtered.groupby(["depmap_id", "drug_name"], as_index=False).agg(lnIC50=("lnIC50", "median"), smiles=("smiles", "first")))

# Save to .csv
gdsc_filtered.to_csv(PROCESSED_DIR / "Response_processed.csv", index=False)
expression.to_csv(PROCESSED_DIR / "expression.csv", index=False)
mrna.to_csv(PROCESSED_DIR / "methylation.csv", index=False)
mutations.to_csv(PROCESSED_DIR / "mutations.csv", index=False)
cn.to_csv(PROCESSED_DIR / "copy_number.csv", index=False)
if use_crispr:
    crispr.to_csv(PROCESSED_DIR / "crispr.csv", index=False)

print('Saved')