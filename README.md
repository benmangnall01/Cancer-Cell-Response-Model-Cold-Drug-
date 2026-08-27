# Cancer-Cell-Response-Model-Cold-Drug-
Code and required data sets to run cancer-cell response model. 

Achieves a mean Pearson correlation of 0.569 in a cold drug environment across a 5-fold cv. 

See 'Scripts' branch for code. \
See 'Raw data sets' branch for raw data used to get model features and run the model. \
See 'Processed data sets' for data sets used to run the model.

Cell features: Gene expression, gene mutation, cell methylation, cell copy number and cell CRISPR importance. \
Drug features: Morgan fingerprints, ChemBERTa embeddings, molecular graph embeddings and drug-target interaction estimated via 'DeepPurpose'.

Uses the machine learning architecture inspired by: Xia, X., Zhu, C., Zhong, F. et al. TransCDR: a deep learning model for enhancing the generalizability of drug activity prediction through transfer learning and multimodal data fusion. BMC Biol 22, 227 (2024). https://doi.org/10.1186/s12915-024-02023-8
