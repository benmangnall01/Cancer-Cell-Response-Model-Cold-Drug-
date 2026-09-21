# TransCDR Reproduction and De Novo Drug Screening

This repository recreates a multimodal TransCDR-style model for predicting cancer cell-line response to small molecules. The model predicts natural-log IC50 (`ln_ic50`) from molecular representations of a drug and genomic features of a cell line.

The project supports two related workflows:

1. Evaluate response prediction with leakage-aware cold-drug or cold-cell cross-validation.
2. Train and save a model, generate novel molecular structures, and rank those structures by predicted response across selected cell lines.

## Model inputs

Each drug is represented by:

- a 1,024-bit Morgan fingerprint;
- a ChemBERTa embedding;
- a molecular-graph embedding produced with DGL-LifeSci; and
- optionally, a drug-target interaction (DTI) embedding.

Each cell line is represented by:

- gene expression;
- somatic mutation status;
- mRNA features;
- copy-number features; and
- optionally, CRISPR gene-effect features.

The modality-specific encoders project these inputs into a shared latent space. A transformer encoder combines the drug and cell-line representations, and a regression head predicts `ln_ic50`. Lower predicted values indicate greater predicted sensitivity.

## Project structure

```text
CDR_Model/
├── raw/                            # Source response, omics, and mapping data
├── processed/                      # Prepared response table and feature caches
├── outputs/                        # Splits, metrics, checkpoints, and screening results
└── scripts/
    ├── Get_cell_features.py        # Prepare response and cell-line features
    ├── Get_drug_features.py        # Build drug feature caches
    ├── Run_model_CV.py             # Five-fold cold-drug or cold-cell evaluation
    ├── Train_full_model.py         # Train and save the model used for screening
    ├── Generate_Drugs.py           # Generate novel SMILES and their features
    ├── Evaluate_generated_drugs.py # Predict and summarize generated-drug responses
    ├── model.py                    # Data loading, model definition, training, and evaluation
    └── model_helper.py             # Transformer building blocks
```

All scripts resolve paths from the project root, so they can be run from the `scripts` directory without editing working-directory paths.

## Data

- [Raw data](https://doi.org/10.5281/zenodo.22095665)
- [Processed data](https://doi.org/10.5281/zenodo.22099989)

Place downloaded data in the corresponding `raw/` or `processed/` directory shown above. If the processed response table and feature caches are already available, the preparation steps can be skipped.

## Environment

Activate the project's `chem_env` environment before running the pipeline. The core dependencies include PyTorch, pandas, NumPy, scikit-learn, SciPy, lifelines, RDKit, Transformers, DGL, DGL-LifeSci, tqdm, and requests. DeepPurpose is additionally required when DTI features are enabled.

The first use of ChemBERTa, the molecular generator, or pretrained graph components may require internet access to download model weights. Their revisions are pinned in the scripts for reproducibility.

## Usage

Run the following commands from `TransCDR_recreate/scripts`.

### 1. Prepare the response and cell-line data

```bash
python Get_cell_features.py
```

This script harmonizes the response and cell-line identifiers, canonicalizes drug SMILES, and writes:

- `processed/Response_processed.csv`
- `processed/expression.csv`
- `processed/mrna.csv`
- `processed/mutations.csv`
- `processed/copy_number.csv`
- `processed/crispr.csv` when CRISPR features are enabled

### 2. Build drug features

```bash
python Get_drug_features.py
```

This produces the fingerprint, ChemBERTa, and molecular-graph caches used by training. DTI features are optional and disabled by default.

### 3. Run cross-validation

```bash
python Run_model_CV.py
```

Set `SCENARIO` in the script to either:

- `cold-drug`: split by canonical SMILES so a structure cannot appear in more than one partition; or
- `cold-cell`: split by DepMap cell-line identifier.

The script performs five-fold evaluation. Within each fold, the held-in data are divided into training and validation partitions for early stopping. Split files and fold metrics are written to:

```text
outputs/cv_splits_<scenario>/
├── train1.csv ... train5.csv
├── val1.csv   ... val5.csv
├── test1.csv  ... test5.csv
└── metrics.csv
```

### 4. Train and save the screening model

```bash
python Train_full_model.py
```

This creates one entity-level 90/5/5 train-validation-test split for the selected scenario. The model is trained on `pairs_train`, validation MSE controls early stopping, and the held-out test partition is used for final evaluation. The selected checkpoint is saved as:

```text
outputs/Models/CDR_model_<scenario>.pt
```

The checkpoint contains the learned weights together with the feature schemas, normalization state, drug-feature dimensions, and feature-use flags required for compatible inference.

### 5. Generate de novo drug candidates

```bash
python Generate_Drugs.py
```

The generator samples SMILES from a pretrained molecular language model, removes invalid and duplicate structures, and excludes molecules already present in the training response data. It then creates the same drug representations used by the response model.

Generation settings such as `N_TO_SAMPLE`, `MAX_LEN`, `TEMPERATURE`, `TOP_K`, and `GEN_BATCH` are defined near the top of the script. Outputs include:

```text
outputs/generated_drugs.csv
outputs/generated_drug_fingerprints.npz
outputs/generated_drug_chemberta_embeddings.npz
outputs/generated_drug_molecular_graphs.npz
outputs/generated_drug_dti.npz              # only when DTI is enabled
```

### 6. Score and rank generated candidates

```bash
python Evaluate_generated_drugs.py
```

Set `MODEL_PATH` to the checkpoint to evaluate and set `TARGET_CELLS` to a list of DepMap IDs, or to `"all"`. The script verifies that the checkpoint, cell features, and generated-drug features are compatible before inference. It also calculates QED as a basic drug-likeness descriptor.

The evaluation writes:

- `outputs/generated_drug_predictions_full.csv`: one predicted `ln_ic50` value for every generated-drug/cell-line pair;
- `outputs/generated_drug_predictions_summary.csv`: per-drug mean and minimum predicted response, the most sensitive predicted cell line, and QED.

Candidates can be prioritized using low predicted `ln_ic50`, with QED and other downstream filters used as complementary criteria rather than evidence of efficacy.

## Configuration notes

- Keep the CRISPR setting consistent between `Get_cell_features.py` and `model.py`.
- If DTI features are enabled, enable them consistently in drug-feature generation, model training, and generated-drug feature generation. A compatible pretrained DTI model is also required.
- `Run_model_CV.py` is intended for performance estimation. `Train_full_model.py` produces the checkpoint consumed by the generated-drug evaluation workflow.
- The default random seed is fixed where sampling or splitting is performed, but GPU operations and library versions can still introduce small run-to-run differences.

## Reference results

The original project configuration reported the following mean Pearson correlations across five folds:

- cold-drug prediction: **0.569**
- cold-cell prediction: **0.886**

Results depend on the exact input-data versions, feature settings, split scenario, and software environment.

## Acknowledgement

This implementation is based on the model described in [TransCDR: a deep learning model for enhancing the generalizability of drug activity prediction through transfer learning and multimodal data fusion](https://doi.org/10.1186/s12915-024-02023-8).
