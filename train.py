import scanpy as sc
import pandas as pd
import anndata
import numpy as np
import torch
from falm import Train_FALM
from preprocess import pca, pca_with_params, clr_normalize_each_cell, construct_neighbor_graph
from preprocess import fix_seed

fix_seed(2025)

print("Loading data...")
adata_rna = sc.read_h5ad("./Datasets/Human_Tonsil/rna.h5ad")
adata_protein = sc.read_h5ad("./Datasets/Human_Tonsil/protein.h5ad")
adata_rna.var_names_make_unique()
adata_protein.var_names_make_unique()

locs = pd.read_csv("./Datasets/Human_Tonsil/locs.csv", index_col=0)
common_barcodes = adata_rna.obs_names.intersection(locs.index)
adata_rna = adata_rna[common_barcodes].copy()
adata_protein = adata_protein[common_barcodes].copy()
locs_filtered = locs.loc[common_barcodes]

spatial_corrds = locs_filtered[['2', '3']].values
adata_rna.obsm['spatial'] = spatial_corrds
adata_protein.obsm['spatial'] = spatial_corrds

print(f"RNA shape: {adata_rna.shape}, Protein shape: {adata_protein.shape}, Spatial coords shape: {spatial_corrds.shape}")

img_emb = np.load('./Datasets/Human_Tonsil/image_embeddings.npy')
print(f"Image embeddings shape: {img_emb.shape}")
adata_img = anndata.AnnData(
    X=None,
    obs=adata_rna.obs.copy(),
    var=adata_rna.var.copy(),
)
adata_img.obsm['feat'] = img_emb.copy()
adata_img.obsm['spatial'] = spatial_corrds.copy()

print("Preprocessing RNA data...")
n_protein = adata_protein.n_vars
sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']].copy()
adata_rna.uns['hvg_genes'] = adata_rna_high.var_names.tolist()
X_hvg = adata_rna_high.X
if hasattr(X_hvg, "toarray"):
    X_hvg = X_hvg.toarray()
X_hvg = X_hvg.astype(np.float32)
adata_rna.obsm['X_hvg'] = X_hvg

feat_rna, comp_rna, mean_rna = pca_with_params(adata_rna_high, n_comps=n_protein, random_state=2025)
adata_rna.obsm['feat'] = feat_rna
adata_rna.uns['pca_rna'] = {"components": comp_rna, "mean": mean_rna}
top_k_gene = 200
gene_blocks = []
for pc in range(comp_rna.shape[0]):
    idx = np.argsort(np.abs(comp_rna[pc]))[::-1][:top_k_gene]
    gene_blocks.append(idx.astype(np.int64))
adata_rna.uns['pca_gene_blocks'] = gene_blocks
adata_rna.uns['pca_gene_blocks_top_k'] = int(top_k_gene)

print("Preprocessing protein data...")
adata_protein = clr_normalize_each_cell(adata_protein)
X_protein = adata_protein.X
if hasattr(X_protein, "toarray"):
    X_protein = X_protein.toarray()
X_protein = X_protein.astype(np.float32)
adata_protein.obsm['X_protein'] = X_protein
feat_adt, comp_adt, mean_adt = pca_with_params(adata_protein, n_comps=n_protein, random_state=2025)
adata_protein.obsm['feat'] = feat_adt
adata_protein.uns['pca_adt'] = {"components": comp_adt, "mean": mean_adt}
top_k_protein = 10
protein_blocks = []
for pc in range(comp_adt.shape[0]):
    idx = np.argsort(np.abs(comp_adt[pc]))[::-1][:top_k_protein]
    protein_blocks.append(idx.astype(np.int64))
adata_protein.uns['pca_protein_blocks'] = protein_blocks
adata_protein.uns['pca_protein_blocks_top_k'] = int(top_k_protein)

print("Constructing neighbor graph...")
data = construct_neighbor_graph(adata_rna, adata_protein, adata_img)

print("Training FALM model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Train_FALM(data, device=device, epochs=600)
torch.autograd.set_detect_anomaly(True)
output = model.train()

print("Saving loss curves...")
model.loss_logger.plot(save_dir="./loss_curves")

print("Saving results...")
adata = adata_rna.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['emb_latent_omics3'] = output['emb_latent_omics3'].copy()
adata.obsm['FALM'] = output['FALM'].copy()

adata.write("./Results/Human_Tonsil/Result.h5ad")

locs_filtered.to_csv("./Results/Human_Tonsil/locs_filtered.csv")

print("Training complete! Results saved to ./Results/Human_Tonsil/")

from evaluate_with_pseudo_gt import evaluate_with_pseudo_gt_from_emb

emb = adata.obsm['FALM']  

pseudo_gt_path = "./pseudo_gt_Human_Tonsil/pseudo_gt_Human_Tonsil.npz"

results = evaluate_with_pseudo_gt_from_emb(
    emb=emb,
    pseudo_gt_path=pseudo_gt_path,            
    gt_keys=None,             
    random_state=2025,
)