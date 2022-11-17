datasets = ["10Xmouse", "humanpancreas", "10Xpbmc"]


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


for RANDOM_SEED in range(100,105):

    results = list()

    for dataset in datasets:
        batch_labels = "batch"
        cell_labels = "actual"
        #a = np.array([0.1* x for x in range(1,5)])
        #b = np.array([0.1* x for x in range(6,10)])
        b = np.array([0.25, 0.5, 0.75])
        a = np.array([0.001, 0.01, 0.1])
        hyper_parameter_space = np.concatenate((a, b), axis=None)
        for l in hyper_parameter_space:
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
            model = IHPF.run_trials(
                Xlist,
                no_cell_types,
                random_seed=RANDOM_SEED,
                ntrials=1,
                max_iter=500,
                model_kwargs={"dataset_ratio": l},
            )
            ## Write to results 
            adata = sc.read("../Data/Finalised_Data/{}.h5ad".format(dataset))    
            adata.obsm["IHPF_{}".format(l)] = np.concatenate(model.cell_scores(), axis=0)
            adata.varm["IHPF_{}".format(l)] = model.shared_gene_scores()
            kmeans_cell = KMeans(n_clusters=no_cell_types, random_state=0).fit(
                normalize(adata.obsm["IHPF_{}".format(l)])
            )
            adata.obs["IHPF_{}_kmeans_normalised".format(l)] = kmeans_cell.labels_
            adata.obs["IHPF_{}_max".format(l)] = np.argmax(
                adata.obsm["IHPF_{}".format(l)], axis=1
            )
            adata.write("../Data/Finalised_Data/{}.h5ad".format(dataset))
            ## Calculate AMI 

            cell_AMI = adjusted_mutual_info_score(adata.obs[cell_labels],adata.obs["IHPF_{}_kmeans_normalised".format(l)])
            batch_AMI = adjusted_mutual_info_score(adata.obs[batch_labels],adata.obs["IHPF_{}_kmeans_normalised".format(l)])
            cell_SC = silhouette_score(adata.obsm["IHPF_{}".format(l)],adata.obs["IHPF_{}_kmeans_normalised".format(l)]) 

            record = {"dataset": dataset, "llh": model.loss[-1], "noise_ratio": l, 'cell_AMI':cell_AMI, 'batch_AMI':batch_AMI, 'cell_SC': cell_SC, 'seed':RANDOM_SEED}
            results.append(record)
            pd.DataFrame.from_records(results).to_csv(
                f"IHPF_noise_ratio_finalised_{RANDOM_SEED}.csv", index=False
            )
