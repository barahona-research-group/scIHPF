datasets = ["humanpancreas", "10Xmouse", "10Xpbmc"]

## Import scripts for the dimension reduction methods
import IHPF
from INMF import INMF
import schpf
from sklearn.decomposition import PCA

from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from sklearn.metrics import adjusted_mutual_info_score, silhouette_score


class scINMF:
    def __init__(self, k, alpha=1, **kwargs):
        self.n_components = k
        self.method = INMF(
            n_components=self.n_components, solver="mu", alpha=alpha, **kwargs
        )

    def fit(self, X):
        self.data = X
        (
            self.cell_scores,
            self.shared_gene_scores,
            self.dataset_gene_scores,
        ) = self.method.fit_transform(self.data)

        
        
for RANDOM_SEED in range(5,20): 
    
    results = list()

    for dataset in datasets:
        batch_labels = "batch"
        cell_labels = "actual"
        ## Replace this with the location of your h5ad files

        hyper_parameter_space = [0.1,1]
        for l in hyper_parameter_space:
            adata = sc.read("../Data/Finalised_Data/{}_data_only.h5ad".format(dataset))
            no_cell_types = len(adata.obs[cell_labels].unique())
            no_batches = len(adata.obs[batch_labels].unique())
            # Split cell matrix into batches preserving order
            Xlist = list()
            split_idx = list()
            for i, df in adata.obs.groupby(batch_labels):
                df_ints = [int(x) for x in df.index]
                split_idx.append(min(df_ints))
            split_idx.append(adata.obs.shape[0])
            split_idx = sorted(split_idx)
            split_starts = split_idx[:-1]
            split_ends = split_idx[1:]
            for i in range(0, no_batches):
                Xlist.append(coo_matrix(adata.X[split_starts[i] : split_ends[i], :]))
            model = scINMF(no_cell_types, alpha=1 / l, max_iter=500, random_state=RANDOM_SEED)
            model.fit(Xlist)

            ## Write to results 
            adata = sc.read("../Data/Finalised_Data/{}.h5ad".format(dataset))  
            adata.obsm["INMF_{}".format(l)] = np.concatenate(model.cell_scores, axis=0)
            adata.varm["INMF_{}".format(l)] = model.shared_gene_scores.transpose()
            kmeans_cell = KMeans(n_clusters=no_cell_types, random_state=0).fit(
                normalize(adata.obsm["INMF_{}".format(l)])
            )
            adata.obs["INMF_{}_kmeans_normalised".format(l)] = kmeans_cell.labels_
            adata.obs["INMF_{}_max".format(l)] = np.argmax(
                adata.obsm["INMF_{}".format(l)], axis=1
            )
            adata.write("../Data/Finalised_Data/{}.h5ad".format(dataset))
            cell_AMI = adjusted_mutual_info_score(adata.obs[cell_labels],adata.obs["INMF_{}_kmeans_normalised".format(l)])
            batch_AMI = adjusted_mutual_info_score(adata.obs[batch_labels],adata.obs["INMF_{}_kmeans_normalised".format(l)])
            cell_SC = silhouette_score(adata.obsm["INMF_{}".format(l)],adata.obs["INMF_{}_kmeans_normalised".format(l)]) 
            record = {"dataset": dataset, "noise_ratio": l, 'cell_AMI':cell_AMI, 'batch_AMI':batch_AMI, 'cell_SC': cell_SC, }
            results.append(record)
            pd.DataFrame.from_records(results).to_csv(
                f"INMF_noise_ratio_finalised_{RANDOM_SEED}.csv", index=False
            )