# Cancer-Cell-Response-Model-Cold-Drug-

A multimodal deep learning model for predicting cancer cell-line drug response from drug molecular structure and cell-line molecular profiles.

The model integrates multiple representations of both the drug and cancer cell line, using pretrained molecular models and a Transformer-based fusion architecture to predict the logarithm of the half-maximal inhibitory concentration (lnIC50).


**Overview**

Predicting how cancer cells respond to different drugs is an important problem in computational oncology and drug discovery. Drug response depends on both the molecular properties of a compound and the biological characteristics of the cancer cell.

This project combines these two sources of information into a single multimodal model.


**Drug representation**

Each drug is represented using three complementary molecular representations:

Morgan fingerprint — a 1,024-bit circular molecular fingerprint generated from the SMILES structure.\
ChemBERTa embedding — a pretrained transformer embedding generated directly from the drug SMILES.\
Molecular graph embedding — a pretrained graph neural network representation generated using DGL-LifeSci.


**Cell-line representation**

Each cancer cell line is represented using five molecular profiles:

Gene expression. \
Somatic mutations. \
DNA methylation. \
Copy number. \
CRISPR gene-dependency data

Each modality is independently projected into a shared 256-dimensional representation.

**Usage**

**1. Prepare the cell-line data**

Cell_features.py

This produces the processed cell-line and drug-response datasets required by the model.

**2. Generate drug features**

Drug_features.py

This generates and caches the three molecular representations:

drug_fingerprints.npz \
drug_chemberta_embeddings.npz \
drug_molecular_graphs.npz

**3. Run cross-validation**

Run_model.py

This will:

Create five drug-level cross-validation folds. \
Create training, validation and test datasets for each fold. \
Train a new model for each fold. \
Select the best model using validation MSE. \
Evaluate the model on the held-out drugs. 

Achieves a mean Pearson correlation of 0.569 in a cold-drug scenario across a 5-fold CV. 


Uses the machine learning architecture inspired by: Xia, X., Zhu, C., Zhong, F. et al. TransCDR: a deep learning model for enhancing the generalizability of drug activity prediction through transfer learning and multimodal data fusion. BMC Biol 22, 227 (2024). https://doi.org/10.1186/s12915-024-02023-8
