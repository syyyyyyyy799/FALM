import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 

def _ensure_csr(x):
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(np.asarray(x))

def construct_hypergraph_matrix_from_data(
    adata,
    km: int = 30,
    w_full: float = 0.5,
    use_highly_variable: bool = True,
    eps: float = 1e-8,
):
    if use_highly_variable and ("highly_variable" in adata.var.columns):
        feat_mask = adata.var["highly_variable"].values.astype(bool)
        X = adata[:, feat_mask].X
    else:
        X = adata.X
    X = _ensure_csr(X)
    n_spot, n_feat = X.shape
    col_sum = np.asarray(X.sum(axis=0)).ravel().astype(np.float32) + eps
    inv_col = 1.0 / col_sum
    A_data = []
    A_rows = []
    A_cols = []
    X = X.tocsr()
    for i in range(n_spot):
        start, end = X.indptr[i], X.indptr[i + 1]
        cols = X.indices[start:end]
        vals = X.data[start:end].astype(np.float32)
        if cols.size == 0:
            continue
        vals_norm = vals * inv_col[cols]
        row_sum = float(vals_norm.sum()) + eps
        sal = vals_norm / row_sum
        k = min(km, sal.size)
        if k <= 0:
            continue
        topk_idx = np.argpartition(-sal, kth=k - 1)[:k]
        top_cols = cols[topk_idx]
        top_vals = vals_norm[topk_idx]
        A_rows.append(np.full(top_cols.shape, i, dtype=np.int32))
        A_cols.append(top_cols.astype(np.int32))
        A_data.append(top_vals.astype(np.float32))
    if len(A_data) == 0:
        A_sparse = sp.csr_matrix((n_spot, n_feat), dtype=np.float32)
    else:
        A_sparse = sp.coo_matrix(
            (np.concatenate(A_data), (np.concatenate(A_rows), np.concatenate(A_cols))),
            shape=(n_spot, n_feat),
            dtype=np.float32,
        ).tocsr()
    if w_full <= 0:
        return A_sparse
    X_norm_full = X.multiply(inv_col)
    A = (w_full * X_norm_full) + ((1.0 - w_full) * A_sparse)
    return A.tocsr()
    
def construct_hypergraph_matrix_from_X(
        X, 
        km: int = 30,
        w_full: float = 0.5,
        eps: float = 1e-8,
):
    X = _ensure_csr(X)
    n_spot, n_feat = X.shape
    col_sum = np.asarray(X.sum(axis=0)).ravel().astype(np.float32) + eps
    inv_col = 1.0 / col_sum
    A_data = []
    A_rows = []
    A_cols = []
    X = X.tocsr()
    for i in range(n_spot):
        start, end = X.indptr[i], X.indptr[i + 1]
        cols = X.indices[start:end]
        vals = X.data[start:end].astype(np.float32)
        if cols.size == 0:
            continue
        vals_norm = vals * inv_col[cols]
        row_sum = float(vals_norm.sum()) + eps
        sal = vals_norm / row_sum
        k = min(km, sal.size)
        if k <= 0:
            continue
        topk_idx = np.argpartition(-sal, kth=k - 1)[:k]
        top_cols = cols[topk_idx]
        top_vals = vals_norm[topk_idx]
        A_rows.append(np.full(top_cols.shape, i, dtype=np.int32))
        A_cols.append(top_cols.astype(np.int32))
        A_data.append(top_vals.astype(np.float32))
    if len(A_data) == 0:
        A_sparse = sp.csr_matrix((n_spot, n_feat), dtype=np.float32)
    else:
        A_sparse = sp.coo_matrix(
            (np.concatenate(A_data), (np.concatenate(A_rows), np.concatenate(A_cols))),
            shape=(n_spot, n_feat),
            dtype=np.float32,
        ).tocsr()
    if w_full <= 0:
        return A_sparse
    X_norm_full = X.multiply(inv_col)
    A = (w_full * X_norm_full) + ((1.0 - w_full) * A_sparse)
    return A.tocsr()

def construct_neighbor_graph(
    adata_omics1, adata_omics2, adata_omics3, n_neighbors=3,
    km_rna: int = 30, km_adt: int = 15, w_full: float = 0.5,
    img_k: int = 20, img_mode: str = "connectivity", img_metric: str = "correlation", img_include_self: bool = False,
    ): 

    # construct spatial neighbor graphs
    ################# spatial graph #################
    print(n_neighbors)
    # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    
    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2
    
    # omics3
    cell_position_omics3 = adata_omics3.obsm['spatial']
    adj_omics3 = construct_graph_by_coordinate(cell_position_omics3, n_neighbors=n_neighbors)
    adata_omics3.uns['adj_spatial'] = adj_omics3
    
    ################# feature graph #################
    A_hyper_rna = construct_hypergraph_matrix_from_data(adata_omics1, km=km_rna, w_full=w_full, use_highly_variable=True)
    A_hyper_adt = construct_hypergraph_matrix_from_data(adata_omics2, km=km_adt, w_full=w_full, use_highly_variable=True)
    adata_omics1.obsm["A_hyper"] = A_hyper_rna
    adata_omics2.obsm["A_hyper"] = A_hyper_adt
    feature_graph_omics3 = kneighbors_graph(adata_omics3.obsm['feat'], img_k, mode=img_mode, metric=img_metric, include_self=img_include_self)
    adata_omics3.obsm["adj_feature"] = feature_graph_omics3
 
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2, 'adata_omics3': adata_omics3}
    
    return data

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def pca_with_params(adata, use_reps=None, n_comps=10, random_state=0):
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps, random_state=random_state)
    if use_reps is not None:
        X = adata.obsm[use_reps]
    else:
        X = adata.X
    if isinstance(X, csc_matrix) or isinstance(X, csr_matrix):
        X = X.toarray()
    feat_pca = pca.fit_transform(X).astype(np.float32)
    components = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    return feat_pca, components, mean

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray()  if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def construct_graph_by_feature(adata_omics1, adata_omics2, adata_omics3, k=20, mode= "connectivity", metric="correlation", include_self=False):
    
    """Constructing feature neighbor graph according to expresss profiles"""
    
    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics3=kneighbors_graph(adata_omics3.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2, feature_graph_omics3

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    #print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adata_omics3):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    
    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)
    adj_spatial_omics3 = adata_omics3.uns['adj_spatial']
    adj_spatial_omics3 = transform_adjacent_matrix(adj_spatial_omics3)
    
    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()
    adj_spatial_omics3 = adj_spatial_omics3.toarray()
    
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)
    adj_spatial_omics3 = adj_spatial_omics3 + adj_spatial_omics3.T
    adj_spatial_omics3 = np.where(adj_spatial_omics3>1, 1, adj_spatial_omics3)
    
    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)
    adj_spatial_omics3 = preprocess_graph(adj_spatial_omics3)
    
    ######################################## construct feature graph ########################################
    adj_feature_omics3 = torch.FloatTensor(adata_omics3.obsm['adj_feature'].copy().toarray())
    adj_feature_omics3 = adj_feature_omics3 + adj_feature_omics3.T
    adj_feature_omics3 = np.where(adj_feature_omics3>1, 1, adj_feature_omics3)
    adj_feature_omics3 = preprocess_graph(adj_feature_omics3)
    
    A_hyper_omics1 = adata_omics1.obsm['A_hyper']
    A_hyper_omics2 = adata_omics2.obsm['A_hyper']
    A_hyper_omics1 = sparse_mx_to_torch_sparse_tensor(A_hyper_omics1.tocoo())
    A_hyper_omics2 = sparse_mx_to_torch_sparse_tensor(A_hyper_omics2.tocoo())
    
    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_spatial_omics3': adj_spatial_omics3,
           'adj_feature_omics3': adj_feature_omics3,
           'A_hyper_omics1': A_hyper_omics1,
           'A_hyper_omics2': A_hyper_omics2
           }
    
    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)

    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)

    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   
    
def fix_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    
