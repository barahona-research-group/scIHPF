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






for RANDOM_SEED in range(1,5):
    results = list()
    for dataset in datasets:
        batch_labels = "batch"
        cell_labels = "actual"
        ## Replace this with the location of your h5ad files
        adata = sc.read('../Data/Finalised_Data/{}_data_only.h5ad'.format(dataset))
        for batch_name in adata.obs['batch'].unique():
            train_obs = adata.obs[adata.obs['batch']!=batch_name]
            test_obs = adata.obs[adata.obs['batch']==batch_name]
            train_index = [int(x) for x in train_obs.index]
            test_index = [int(x) for x in test_obs.index]
            train_adata = AnnData(X=adata.X[train_index,:],obs=train_obs.reset_index(),var=adata.var)
            test_adata = AnnData(X=adata.X[test_index,:],obs=test_obs.reset_index(),var=adata.var)
            ## Train on 
            no_cell_types = len(train_adata.obs[cell_labels].unique())
            no_batches = len(train_adata.obs[batch_labels].unique())
            # Split cell matrix into batches preserving order 
            Xlist = list()
            split_idx = list()
            for i,df in train_adata.obs.groupby('batch'):
                if df.shape[0] > 0:
                    df_ints = [int(x) for x in df.index]
                    split_idx.append(min(df_ints))
            split_idx.append(train_adata.obs.shape[0])
            split_idx = sorted(split_idx)
            split_starts = split_idx[:-1]
            split_ends = split_idx[1:]
            for i in range(0,no_batches):
                Xlist.append(coo_matrix(train_adata.X[split_starts[i]:split_ends[i],:]))
            ## Search for Noise Ratio 
            for noise_ratio in [0.001,0.01,0.1,0.25,0.5,0.75,1]:
                model = IHPF.run_trials(Xlist,no_cell_types, random_seed=RANDOM_SEED, max_iter=500, ntrials=1, model_kwargs = {'dataset_ratio':noise_ratio,},verbose=False)
                ## Extract shared gene score for HPF 
                shared_genes = model.shared_gene_scores()
                test_model = schpf.scHPF(nfactors = no_cell_types, bp=np.mean(model.bp), dp=np.mean(model.dp), eta=model.eta, beta=model.beta)
                test_model = test_model.project(coo_matrix(test_adata.X))
                results.append({'dataset':dataset,'batch':batch_name,'noise':noise_ratio,'llh':test_model.loss[-1],'seed':RANDOM_SEED})
                pd.DataFrame(results).to_csv(f'IHPF_batch_crossvalidation_finalised_{RANDOM_SEED}.csv', index=False)