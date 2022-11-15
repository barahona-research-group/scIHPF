datasets = ["humanpancreas", "10Xmouse", "10Xpbmc"]


## Import scripts for the dimension reduction methods
import IHPF
from INMF import INMF
import schpf
from sklearn.decomposition import PCA

from scipy.sparse import coo_matrix, vstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from sklearn.metrics import adjusted_mutual_info_score, silhouette_score


for RANDOM_SEED in range(5,20):

    results = list()

    for dataset in datasets:
        batch_labels = "batch"
        cell_labels = "actual"
        b = np.array([0.25, 0.5, 0.75])
        a = np.array([0.001, 0.01, 0.1])
        hyper_parameter_space = np.concatenate((a, b), axis=None)
        for l in [0]:
            print(dataset, l)
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
            # model = IHPF.scIHPF(no_cell_types,max_iter=500)
            # model.fit(Xlist,dataset_ratio=l)
            # Try different random seeds
            np.random.seed(RANDOM_SEED)
            model = schpf.run_trials(vstack(Xlist), no_cell_types, ntrials=1)
            ## Write to results 
            adata = sc.read("../Data/Finalised_Data/{}.h5ad".format(dataset))    
            adata.obsm['HPF'] = model.cell_score()
            adata.varm['HPF'] = model.gene_score()
            kmeans_cell = KMeans(n_clusters=no_cell_types, random_state=0).fit(normalize(adata.obsm['HPF']))
            adata.obs['HPF_kmeans_normalised'] = kmeans_cell.labels_
            adata.obs['HPF_max'] = np.argmax(adata.obsm['HPF'],axis=1)
            adata.write("../Data/Finalised_Data/{}.h5ad".format(dataset))
            ## Calculate AMI 
            cell_AMI = adjusted_mutual_info_score(adata.obs[cell_labels],adata.obs["HPF_kmeans_normalised"])
            batch_AMI = adjusted_mutual_info_score(adata.obs[batch_labels],adata.obs["HPF_kmeans_normalised"])
            cell_SC = silhouette_score(adata.obsm["HPF"],adata.obs["HPF_kmeans_normalised"]) 
            record = {"dataset": dataset, "llh": model.loss[-1], "noise_ratio": l, 'cell_AMI':cell_AMI, 'batch_AMI':batch_AMI, 'cell_SC': cell_SC, 'seed':RANDOM_SEED}
            results.append(record)
            pd.DataFrame.from_records(results).to_csv(
                f"HPF_noise_ratio_finalised_{RANDOM_SEED}.csv", index=False
            )
