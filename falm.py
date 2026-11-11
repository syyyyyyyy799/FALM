import torch
from tqdm import tqdm
import torch.nn.functional as F
from model import Encoder_overall
from preprocess import adjacent_matrix_preprocessing, construct_hypergraph_matrix_from_X, sparse_mx_to_torch_sparse_tensor
import numpy as np
from utils import LossLogger
import scipy.sparse as sp

def _torch_dense_to_csr(x_torch):
    x = x_torch.detach().cpu().numpy()
    return sp.csr_matrix(x)
class Train_FALM:
    def __init__(self, 
        data,
        datatype = 'Triplet',
        device= torch.device('cpu'),
        random_seed = 2025,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        # 0,1,2: recon; 3: modal-mm; 4: mask-mm
        # weight_factors = [1, 1, 1, 0.01, 0.1, 0.1, 0.15]
        weight_factors = [1, 1, 1, 0.01, 0.1, 0.1, 0]
        ):
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adata_omics3 = self.data['adata_omics3']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, self.adata_omics3)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_spatial_omics3 = self.adj['adj_spatial_omics3'].to(self.device)
        self.A_hyper_omics1 = self.adj['A_hyper_omics1'].to(self.device)
        self.A_hyper_omics2 = self.adj['A_hyper_omics2'].to(self.device)
        self.adj_feature_omics3 = self.adj['adj_feature_omics3'].to(self.device)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.features_omics3 = torch.FloatTensor(self.adata_omics3.obsm['feat'].copy()).to(self.device)

        # RNA
        self.X_hvg_rna = torch.FloatTensor(self.adata_omics1.obsm['X_hvg'].copy()).to(self.device)
        pca_rna = self.adata_omics1.uns['pca_rna']
        self.pca_rna_components = torch.FloatTensor(pca_rna['components'].copy()).to(self.device)
        self.pca_rna_mean = torch.FloatTensor(pca_rna['mean'].copy()).to(self.device)
        self.rna_blocks = self.adata_omics1.uns['pca_gene_blocks']
        self.rna_gene_mean = self.X_hvg_rna.mean(dim=0)
        
        # protein
        self.X_protein = torch.FloatTensor(self.adata_omics2.obsm['X_protein'].copy()).to(self.device)
        pca_adt = self.adata_omics2.uns['pca_adt']
        self.pca_adt_components = torch.FloatTensor(pca_adt['components'].copy()).to(self.device)
        self.pca_adt_mean = torch.FloatTensor(pca_adt['mean'].copy()).to(self.device)
        self.adt_blocks = self.adata_omics2.uns['pca_protein_blocks']
        self.protein_mean = self.X_protein.mean(dim=0)

        # image
        self.img_mean = self.features_omics3.mean(dim=0)

        # mask settings
        self.K_mask_views = 1
        self.rna_blocks_subsample = 0.75
        self.adt_block_subsample = 0.75
        self.fill_with_mean = True
        self.p_img_drop = 0.5
        self.rna_n_mask_blocks = 10
        self.adt_n_mask_blocks = 10
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        self.n_cell_omics3 = self.adata_omics3.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_input3 = self.features_omics3.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        self.dim_output3 = self.dim_output

        self.n_hyper_feat1 = int(self.A_hyper_omics1.shape[1])
        self.n_hyper_feat2 = int(self.A_hyper_omics2.shape[1])

        self.loss_logger = LossLogger()
    
    def _pca_transform(self, X, mean, components):
        X_centered = X - mean
        X_pca = torch.matmul(X_centered, components.T)
        return X_pca

    def _mask_block(self, X, blocks, subsample_ratio, fill_mean_vec, n_blocks: int=1):
        pcs = np.random.choice(len(blocks), size=n_blocks, replace=False)
        # print(len(blocks))    35
        mask_idx_list = []
        for pc in pcs:
            idx = torch.as_tensor(blocks[pc], device=self.device, dtype=torch.long)
            m = max(1, int(len(idx) * subsample_ratio))
            perm = torch.randperm(len(idx), device=self.device)[:m]
            mask_idx_list.append(idx[perm])
        mask_idx = torch.unique(torch.cat(mask_idx_list, dim=0))
        X_masked = X.clone()
        if self.fill_with_mean:
            X_masked[:, mask_idx] = fill_mean_vec[mask_idx]
        else:
            X_masked[:, mask_idx] = 0.0
        return X_masked
    
    def _build_masked_features(self):
        X_rna_m = self._mask_block(self.X_hvg_rna, self.rna_blocks, self.rna_blocks_subsample, self.rna_gene_mean, n_blocks=self.rna_n_mask_blocks)
        feeat1_m = self._pca_transform(X_rna_m, self.pca_rna_mean, self.pca_rna_components)

        X_adt_m = self._mask_block(self.X_protein, self.adt_blocks, self.adt_block_subsample, self.protein_mean, n_blocks=self.adt_n_mask_blocks)
        feeat2_m = self._pca_transform(X_adt_m, self.pca_adt_mean, self.pca_adt_components)

        feat3_m = self.features_omics3
        if np.random.rand() < self.p_img_drop:
            if self.fill_with_mean:
                feat3_m = self.img_mean.unsqueeze(0).expand_as(self.features_omics3).clone()
            else:
                feat3_m = torch.zeros_like(self.features_omics3)
        return feeat1_m, feeat2_m, feat3_m, X_rna_m, X_adt_m


    def train(self):
        self.model = Encoder_overall(
            self.dim_input1, self.dim_output1, 
            self.dim_input2, self.dim_output2, 
            self.dim_input3, self.dim_output3,
            n_hyper_feat_omics1 = self.n_hyper_feat1,
            n_hyper_feat_omics2 = self.n_hyper_feat2,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(
                self.features_omics1, self.features_omics2, self.features_omics3,
                self.adj_spatial_omics1, self.A_hyper_omics1,
                self.adj_spatial_omics2, self.A_hyper_omics2,
                self.adj_spatial_omics3, self.adj_feature_omics3,  
            )

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            self.loss_recon_omics3 = F.mse_loss(self.features_omics3, results['emb_recon_omics3'])

            # loss between modal and multimodal
            h1 = results['emb_latent_omics1']  # RNA
            h2 = results['emb_latent_omics2']  # ADT
            h3 = results['emb_latent_omics3']  # IMG
            h_mm = results['emb_latent_combined']  # multimodal
            z1 = results['z_omics1']
            z2 = results['z_omics2']    
            z3 = results['z_omics3']
            z_mm = results['z_combined']

            loss_pair_rna_adt = 0.5 * (
                masked_info_nce_loss(h1, h2) + masked_info_nce_loss(h2, h1)
            )
            loss_pair_rna_img = 0.5 * (
                masked_info_nce_loss(h1, h3) + masked_info_nce_loss(h3, h1)
            )
            loss_pair_adt_img = 0.5 * (
                masked_info_nce_loss(h2, h3) + masked_info_nce_loss(h3, h2)
            )
            # 三对取平均作为总的 pairwise loss
            loss_pair = (loss_pair_rna_adt + loss_pair_rna_img + loss_pair_adt_img) / 3.0
            # graph smoothness loss
            loss_smooth = (
                graph_smoothness_loss(h1, self.adj_spatial_omics1) +
                graph_smoothness_loss(h2, self.adj_spatial_omics2) +
                graph_smoothness_loss(h3, self.adj_spatial_omics3)
            ) / 3.0
            # loss between uni-modal and multimodal
            loss_uni_rna = barlow_twins_loss(z1, z_mm)
            loss_uni_adt = barlow_twins_loss(z2, z_mm)
            loss_uni_img = barlow_twins_loss(z3, z_mm)
            loss_modal_mm = (loss_uni_rna + loss_uni_adt + loss_uni_img) / 3.0

            loss_mask_mm = 0

            K = self.K_mask_views
            for _ in range(K):
                feat1_m, feat2_m, feat3_m, X_rna_m, X_adt_m = self._build_masked_features()
                X_rna_csr = sp.csr_matrix(X_rna_m.detach().cpu().numpy())
                X_adt_csr = sp.csr_matrix(X_adt_m.detach().cpu().numpy())
                A_rna_mask = construct_hypergraph_matrix_from_X(X_rna_csr, km=30, w_full=0.5)
                A_adt_mask = construct_hypergraph_matrix_from_X(X_adt_csr, km=30, w_full=0.5)
                A_rna_mask_t = sparse_mx_to_torch_sparse_tensor(A_rna_mask.tocoo()).to(self.device)
                A_adt_mask_t = sparse_mx_to_torch_sparse_tensor(A_adt_mask.tocoo()).to(self.device)
                results_m = self.model(
                    feat1_m, feat2_m, feat3_m,
                    self.adj_spatial_omics1, A_rna_mask_t,
                    self.adj_spatial_omics2, A_adt_mask_t,
                    self.adj_spatial_omics3, self.adj_feature_omics3,  
                )
                z_mm_m = results_m['z_combined']
                loss_mask_mm += barlow_twins_loss(z_mm_m, z_mm)
            loss_mask_mm = loss_mask_mm / K

            loss = self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2 + self.weight_factors[2] * self.loss_recon_omics3 \
                 + self.weight_factors[3] * loss_modal_mm + self.weight_factors[4] * loss_mask_mm + self.weight_factors[5] * loss_pair + self.weight_factors[6] * loss_smooth

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step() 
            self.loss_logger.step({
                "recon_omics1": self.loss_recon_omics1.item(),
                "recon_omics2": self.loss_recon_omics2.item(),
                "recon_omics3": self.loss_recon_omics3.item(),
                "modal_mm":     loss_modal_mm.item(),
                "mask_mm":      loss_mask_mm.item(),
                "pairwise":     loss_pair.item(),
                "smoothness":   loss_smooth.item(),
            }, epoch)

        with torch.no_grad():
          self.model.eval()
          results = self.model(
            self.features_omics1, self.features_omics2, self.features_omics3, 
            self.adj_spatial_omics1, self.A_hyper_omics1, 
            self.adj_spatial_omics2, self.A_hyper_omics2, 
            self.adj_spatial_omics3, self.adj_feature_omics3
        )
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_omics3 = F.normalize(results['emb_latent_omics3'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_latent_omics3': emb_omics3.detach().cpu().numpy(),
                  'FALM': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha_omics3': results['alpha_omics3'].detach().cpu().numpy(),
                }
        return output
    
import torch
import torch.nn.functional as F

def masked_info_nce_loss(z1, z2, mask=None, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()  

    batch_size = z1.size(0)
    labels = torch.arange(batch_size).long().to(z1.device)

    if mask is not None:
        logits = logits.masked_fill(~mask, float('-inf'))

    loss = F.cross_entropy(logits, labels)
    return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def barlow_twins_loss(z1, z2, lambd=0.005, eps=1e-12):
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + eps)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + eps)

    N, D = z1.size()
    c = torch.matmul(z1.T, z2) / N  

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    return on_diag + lambd * off_diag

def graph_smoothness_loss(h, adj_sparse):
    Ah = torch.sparse.mm(adj_sparse, h)
    return ((h - Ah) ** 2).sum(dim=1).mean()

from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_mask(z, threshold=0.8):
    z_cpu = z.detach().cpu().numpy()
    sim = cosine_similarity(z_cpu)
    mask = sim < threshold 
    np.fill_diagonal(mask, True)  
    return torch.tensor(mask, dtype=torch.bool, device=z.device)