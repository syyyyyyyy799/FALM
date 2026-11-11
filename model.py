import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from fasterkan import FasterKANLayer

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x, adj):
        N, d = x.shape
        device = x.device

        Q = self.W_q(x)
        K_ = self.W_k(x)
        V = self.W_v(x)
        att_logits = torch.matmul(Q, K_.T) / (d ** 0.5)
        
        if adj.layout == torch.sparse_coo:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        mask = adj_dense > 0
        if mask.dim() == 2:
            mask = mask.clone()
            diag_idx = torch.arange(N, device=device)
            mask[diag_idx, diag_idx] = True
        att_logits = att_logits.masked_fill(~mask, float('-inf'))
        att_weights = torch.softmax(att_logits, dim=-1)

        if torch.isnan(att_weights).any():
            att_weights = torch.nan_to_num(att_weights, nan=0.0)

        out = torch.matmul(att_weights, V)
        out = out + x
        return out

class HypergraphEncoder(nn.Module):
    def __init__(self, n_feat: int, out_feat: int, dropout: float = 0.0):
        super().__init__()
        self.n_feat = int(n_feat)
        self.out_feat = int(out_feat)
        self.dropout = dropout
        self.feat_emb = nn.Parameter(torch.empty(self.n_feat, self.out_feat))
        self.norm = nn.LayerNorm(self.out_feat)
        self.ffn = nn.Sequential(
            nn.Linear(self.out_feat, self.out_feat),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_feat, self.out_feat),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.feat_emb)

    def forward(self, A_hyper):
        if A_hyper.is_sparse:
            z = torch.sparse.mm(A_hyper, self.feat_emb)
        else:
            z = A_hyper @ self.feat_emb
        z = self.norm(z)
        z = self.ffn(z) + z
        return z

class Encoder_overall(Module):
    def __init__(
        self, 
        dim_in_feat_omics1, dim_out_feat_omics1, 
        dim_in_feat_omics2, dim_out_feat_omics2, 
        dim_in_feat_omics3, dim_out_feat_omics3, 
        n_hyper_feat_omics1,
        n_hyper_feat_omics2,
        dropout=0.0, act=F.relu
        ):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_in_feat_omics3 = dim_in_feat_omics3
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dim_out_feat_omics3 = dim_out_feat_omics3
        self.dropout = dropout
        self.act = act
        
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        self.encoder_omics3 = Encoder(self.dim_in_feat_omics3, self.dim_out_feat_omics3)
        self.decoder_omics3 = Decoder(self.dim_out_feat_omics3, self.dim_in_feat_omics3)

        self.hyper_omics1 = HypergraphEncoder(n_hyper_feat_omics1, self.dim_out_feat_omics1, dropout=self.dropout)
        self.hyper_omics2 = HypergraphEncoder(n_hyper_feat_omics2, self.dim_out_feat_omics2, dropout=self.dropout)

        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_omics3 = AttentionLayer(self.dim_out_feat_omics3, self.dim_out_feat_omics3)  
        self.qformer = QFormer(dim=self.dim_out_feat_omics1, num_quiried=4, num_heads=4, num_layers=1, dropout=0.1)
        self.mlp = nn.Sequential(nn.Linear(self.dim_out_feat_omics1 + self.dim_out_feat_omics2 + self.dim_out_feat_omics3, dim_out_feat_omics1))                  

        proj_dim = self.dim_out_feat_omics1
        self.proj_omics1 = nn.Sequential(nn.Linear(self.dim_out_feat_omics1, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.proj_omics2 = nn.Sequential(nn.Linear(self.dim_out_feat_omics2, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.proj_omics3 = nn.Sequential(nn.Linear(self.dim_out_feat_omics3, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.proj_combined = nn.Sequential(nn.Linear(self.dim_out_feat_omics1, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))

    def forward(
        self, 
        features_omics1, features_omics2, features_omics3, 
        adj_spatial_omics1, A_hyper_omics1, 
        adj_spatial_omics2, A_hyper_omics2, 
        adj_spatial_omics3, adj_feature_omics3
        ):

        # graph1
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
        emb_latent_spatial_omics3 = self.encoder_omics3(features_omics3, adj_spatial_omics3)
        
        # graph2
        emb_latent_feature_omics1 = self.hyper_omics1(A_hyper_omics1)
        emb_latent_feature_omics2 = self.hyper_omics2(A_hyper_omics2)
        emb_latent_feature_omics3 = self.encoder_omics3(features_omics3, adj_feature_omics3)

        # within-modality attention aggregation layer
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1) #RNA
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2) #ADT
        emb_latent_omics3, alpha_omics3 = self.atten_omics3(emb_latent_spatial_omics3, emb_latent_feature_omics3) #IMG

        emb_latent_combined, q_tokens = self.qformer(emb_latent_omics1, emb_latent_omics2, emb_latent_omics3)

        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)
        emb_recon_omics3 = self.decoder_omics3(emb_latent_combined, adj_spatial_omics3)
        
        z_omics1 = self.proj_omics1(emb_latent_omics1)
        z_omics2 = self.proj_omics2(emb_latent_omics2)
        z_omics3 = self.proj_omics3(emb_latent_omics3)
        z_combined = self.proj_combined(emb_latent_combined)

        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_omics3':emb_latent_omics3,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'emb_recon_omics3':emb_recon_omics3,
                   'alpha_omics1':alpha_omics1,
                   'alpha_omics2':alpha_omics2,
                   'alpha_omics3':alpha_omics3,
                   'z_omics1': z_omics1,
                   'z_omics2': z_omics2,
                   'z_omics3': z_omics3,
                   'z_combined': z_combined,
                   }
        
        return results
    
class Encoder(Module): 
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.kan = FasterKANLayer(input_dim=out_feat, output_dim=out_feat)
        self.attn = Attention(dim=out_feat)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj, visualize=False):
        x = torch.mm(feat, self.weight)
        x = self.kan(x)   
        x = self.attn(x, adj)
        return x

    
class Decoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.kan = FasterKANLayer(input_dim=out_feat, output_dim=out_feat)
        self.reset_parameters()
        self.attn = Attention(dim=out_feat)
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)                 
        x = self.kan(x)
        x = self.attn(x, adj)
        return x                  

class AttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha
    
    
from zeta.nn import FeedForward     

class QFormer(nn.Module):
    def __init__(self, dim, num_quiried: int = 4, num_heads: int = 4, num_layers: int = 1, dim_feedforward: int = None, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_quiried = num_quiried
        if dim_feedforward is None:
            dim_feedforward = dim * 4
        self.query = nn.Parameter(torch.randn(num_quiried, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, h_rna, h_adt, h_img):
        N, D = h_rna.shape
        queries = self.query.unsqueeze(0).expand(N, self.num_quiried, D)
        model_tokens = torch.stack([h_rna, h_adt, h_img], dim=1)
        tokens = torch.cat([queries, model_tokens], dim=1)
        encoded = self.encoder(tokens)
        q_out = encoded[:, :self.num_quiried, :]
        fused = q_out.mean(dim=1)
        return fused, q_out
    # def forward(self, h_rna, h_adt, h_img):
    #     N, D = h_adt.shape
    #     queries = h_adt.unsqueeze(1)
    #     model_tokens = torch.stack([h_rna, h_img], dim=1)
    #     tokens = torch.cat([queries, model_tokens], dim=1)
    #     encoded = self.encoder(tokens)
    #     q_out = encoded[:, :1, :]
    #     fused = q_out.squeeze(1)
        
    #     return fused, q_out
