import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import scipy
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import kneighbors_graph
from PIL import Image
import random
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def cluster(model, n_clusters=10):
    clusters = KMeans(n_clusters, random_state=100).fit_predict(model.emb)
    model.clusters = clusters
    return clusters

def protein_norm(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)


def preprocess(adata,modality):
  adata.var_names_make_unique()
  if modality in ['rna','atac']:
    sc.pp.filter_genes(adata,min_cells=10)
    sc.pp.log1p(adata)

    if scipy.sparse.issparse(adata.X):
      return adata.X.A
    else:
      return adata.X

  elif modality=='protein':
    adata.X = np.apply_along_axis(protein_norm, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X)))
    return adata.X     

  elif modality=='metabolite':
    sc.pp.log1p(adata)
    if scipy.sparse.issparse(adata.X):
      return adata.X.A
    else:
      return adata.X


def calculate_affinity(X1, sig=30, sparse = False, neighbors = 100):
  if not sparse:
    dist1 = pairwise_distances(X1)
    a1 = np.exp(-1*(dist1**2)/(2*(sig**2)))
    return a1
  else:
    dist1 = kneighbors_graph(X1, n_neighbors = neighbors, mode='distance')
    dist1.data = np.exp(-1*(dist1.data**2)/(2*(sig**2)))
    dist1.eliminate_zeros()
    return dist1

def cmap_tab20(x):
    cmap = plt.get_cmap('tab20')
    x = x % 20
    x = (x // 10) + (x % 10) * 2
    return cmap(x)



def cmap_tab30(x):
    n_base = 20
    n_max = 30
    brightness = 0.7
    brightness = (brightness,) * 3 + (1.0,)
    isin_base = (x < n_base)[..., np.newaxis]
    isin_extended = ((x >= n_base) * (x < n_max))[..., np.newaxis]
    isin_beyond = (x >= n_max)[..., np.newaxis]
    color = (
        isin_base * cmap_tab20(x)
        + isin_extended * cmap_tab20(x-n_base) * brightness
        + isin_beyond * (0.0, 0.0, 0.0, 1.0))
    return color


def cmap_tab70(x):
    cmap_base = cmap_tab30
    brightness = 0.5
    brightness = np.array([brightness] * 3 + [1.0])
    color = [
        cmap_base(x),  
        1 - (1 - cmap_base(x-20)) * brightness,  
        cmap_base(x-20) * brightness, 
        1 - (1 - cmap_base(x-40)) * brightness**2, 
        cmap_base(x-40) * brightness**2,  
        [0.0, 0.0, 0.0, 1.0],  
        ]
    x = x[..., np.newaxis]
    isin = [
        (x < 30),
        (x >= 30) * (x < 40),
        (x >= 40) * (x < 50),
        (x >= 50) * (x < 60),
        (x >= 60) * (x < 70),
        (x >= 70)]
    color_out = np.sum(
            [isi * col for isi, col in zip(isin, color)],
            axis=0)
    return color_out


def plot(clusters,locs):
  locs['2'] = locs['2'].astype('int')
  locs['3'] = locs['3'].astype('int')
  im1 = np.empty((locs['2'].max()+1, locs['3'].max()+1))
  im1[:] = np.nan
  im1[locs['2'],locs['3']] = clusters
  im2 = cmap_tab70(im1.astype('int'))
  im2[np.isnan(im1)] = 1
  im3 = Image.fromarray((im2 * 255).astype(np.uint8))
  return im3

def plot_on_histology(clusters, locs, im, scale, s=10):
  locs = locs*scale
  locs = locs.round().astype('int')
  im = im[(locs['4'].min()-10):(locs['4'].max()+10),(locs['5'].min()-10):(locs['5'].max()+10)]
  locs = locs-locs.min()+10
  cmap1 = mcolors.ListedColormap([cmap_tab70(np.array(i)) for i in range(len(np.unique(clusters)))])
  # plt.imshow(im, alpha=0.7); 
  plot = plt.scatter(x=locs['5'], y=locs['4'], c = clusters, cmap=cmap1, s=s); 
  plt.axis('off'); 
  return plot

def set_random_seed(seed=100):
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)

class LossLogger:
    def __init__(self):
        self.history = {}   # key -> list
        self.epochs = []

    def step(self, loss_dict: dict, epoch: int = None):
        if epoch is None:
            epoch = len(self.epochs)
        self.epochs.append(epoch)
        for k, v in loss_dict.items():
            self.history.setdefault(k, []).append(v)

    def plot(self, save_dir="./", dpi=200):
        import os, matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)
        for name, values in self.history.items():
            plt.figure(figsize=(4, 3))
            plt.plot(self.epochs, values, label=name)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"{name} curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=dpi)
            plt.close()

def relabel_by_overlap_one_to_one(clusters_new, clusters_ref, k=None):
    clusters_new = np.asarray(clusters_new)
    clusters_ref = np.asarray(clusters_ref)
    assert clusters_new.shape == clusters_ref.shape, "clusters_new/ref must have same shape"

    new_labels = np.unique(clusters_new)
    ref_labels = np.unique(clusters_ref)

    if k is None:
        k_new, k_ref = len(new_labels), len(ref_labels)
    else:
        k_new, k_ref = k, k

    if len(new_labels) != len(ref_labels):
        raise ValueError(
            f"Need same number of clusters for 1↔1 mapping, but got "
            f"new={len(new_labels)}, ref={len(ref_labels)}. "
            f"Either fix K to be equal, or use a many-to-one mapping."
        )

    new_to_i = {lab: i for i, lab in enumerate(new_labels)}
    ref_to_j = {lab: j for j, lab in enumerate(ref_labels)}

    M = np.zeros((len(new_labels), len(ref_labels)), dtype=int)
    for cn, cr in zip(clusters_new, clusters_ref):
        M[new_to_i[cn], ref_to_j[cr]] += 1

    cost = -M
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping_new_to_ref = {int(new_labels[i]): int(ref_labels[j]) for i, j in zip(row_ind, col_ind)}

    clusters_aligned = np.array([mapping_new_to_ref[int(c)] for c in clusters_new], dtype=int)

    matched_overlap = int(M[row_ind, col_ind].sum())
    total = int(M.sum())
    info = {
        "overlap_matrix": M,
        "matched_overlap_sum": matched_overlap,
        "total_sum": total,
        "matched_ratio": matched_overlap / max(1, total),
        "mapping_new_to_ref": mapping_new_to_ref,
        "new_labels": new_labels,
        "ref_labels": ref_labels,
    }
    return clusters_aligned, mapping_new_to_ref, info  