 	# Import workhorses
import scipy.io as sio
import scanpy as sc
import numpy as np
#import numba
import pandas as pd 
import anndata as ad
import scipy.sparse 
#from psutil import virtual_memory
import time 
import os
import seaborn as sns


def load_pbmc_markers():

	pbmc_cell_type_markers = {
	'IL2RA' :'T cell',
	'TNFRSF18' :'T cell',
	'CD3D': 'T cell',
	'BATF': 'T helper',
	'IL7R': 'T helper',
	'CD4': 'T helper',
	'GNLY': 'NK T cell',
	'NKG7': 'NK T cell',
	'CD56': 'NK T cell',
	'CCL5': 'Cytotoxic T cell',
	'CD8A': 'Cytotoxic T cell',
	'CD16': 'Cytotoxic T cell',
	'MS4A1': 'B cell',
	'CD79A': 'B cell',
	'LYZ': 'Monocyte',
	'FCGR3A': 'Monocyte',
	'MS4A7': 'Monocyte',
	'CD163': 'Macrophage',
	'GBP1': 'Macrophage',
	'FCER1A': 'Dendritic',
	'LAD1': 'Dendritic',
	'LAMP3': 'Dendritic',
	'FLT3': 'Dendritic',
	'CST3': 'Dendritic',
	'PPBP': 'Megakaryocytes',
	}
	return pbmc_cell_type_markers

def load_cell_type_collapser():
	cell_type_collapser = {
	'B cell': 'B cell', 
	'T cell': 'T cell',
	'Cytotoxic T cell': 'T cell', 
	'T helper': 'T cell',
	'NK T cell': 'T cell', 
	'Macrophage': 'Monocyte', 
	'Monocyte': 'Monocyte', 
	'Dendritic': 'Monocyte',
	'Megakaryocytes': 'Monocyte'
	}
	return cell_type_collapser


def preprocess_scanpy(adata, features, metadata = None, filter_mito = False, min_max_scale = False,
	min_cell_thresh = 5, **kwargs): 
	
	"""
	Custom pipeline to preprocess a single cell RNAseq
	count matrix using scanpy. 

	Params
	------
	adata (ad.AnnData)

	features (pd.DataFrame)

	metadata (pd.DataFrame)

	Returns 
	-------
	adata ()
		Preprocessed anndata. 

	"""

	# Add gene names if the adata is not annotated. 
	if metadata is not None: 
		# Set gene names as var names 
		adata.var_names = features.gene_name.values

		# Eliminate dup var names 
		adata.var_names_make_unique()  

	# Filter cells with less than 300 genes expressed
	# and less than 2500 total UMIs
	sc.pp.filter_cells(adata, min_genes=300)
	sc.pp.filter_cells(adata, min_counts = 2500)

	# Filter genes expressed in less than "min_cell_thresh" cells
	sc.pp.filter_genes(adata, min_cells= min_cell_thresh)
	
	# Add the total counts per cell as observations-annotation to adata
	adata.obs['n_counts'] = adata.X.sum(axis=1).A1

	# Get percentile of lib size distribution 
	perc_low, perc_hi  = np.percentile(adata.obs.n_genes.values, [2.5, 97.5])
	# Keep cells in the 97.5 percentiles of the counts distribution

	adata = adata[adata.obs.n_genes < perc_hi, :]
	adata = adata[adata.obs.n_genes > perc_low, :]

	# Keep cells with less than 10% of mitochondrial genes 
	if filter_mito:
		# Get mitochondrial genes 
		mito_genes = adata.var.index.str.startswith('MT-')

		# For each cell compute fraction of counts in mito genes vs. all genes
		# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
		adata.obs['percent_mito'] = np.sum(
		    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1

		adata = adata[adata.obs.percent_mito < 0.1, :]

	# Normalize by libsize 
	sc.pp.normalize_total(adata, target_sum=1e4)

	# Set data to log scale with a pseudocount
	sc.pp.log1p(adata)

	# Dump raw data 
	adata.raw = adata

	# Feature selection : Get most variable genes (in log scale)
	sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
	adata = adata[:, adata.var.highly_variable]

	# Regress out effects of library size and mitocondrial gene content 
	if filter_mito: 
		
		sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

	# Regress out effects of library size 
	else:
		sc.pp.regress_out(adata, ['n_counts'])


	if min_max_scale: 
		# Min-max scale 
		sc.pp.scale(adata, max_value=1)


	return adata 


def nmf_wrapper_sc(
    data,
    marker_dict,
    n_clusters = 10,
    n_enriched_feats=1,
    feature_names=None, 
    ):
    
    """
    Python wrapper to implement NMF algorithm. 

    It returns the W and H matrices, the cluster_labels per sample, 
    and the top enriched features per cluster. 

    Assumes that either that either sklearn or the 
    sklearn.decomposition.NMF class are loaded in 
    the workspace.

    Params
    -------
    n_clusters (int): 
        Number of components to make the matrix factorization. 

    data: (pd.DataFrame or np.array)
        Dataset to be decomposed. All datapoints need to
        be non-negative. 

    feature_names (array-like): 
        Name of the columns of the dataset.
        This is used to compute the feature enrichment.

    n_enriched_feats (int): 
        number of top enriched features to extract from
        each cluster. In the scRNAseq case, it amounts to get 
        the top genes in each cluster. 

    Returns
    --------
    nmf_W (np.ndarray):
        Cluster matrix, it has (n_samples, n_clusters) shape. 

    nmf_H (np.ndarray): 
        Feature coefficient matrix, it has (n_clusters, n_feats)

    cluster_labels (list):
        Cluster label per sample. 
    
    enriched_feats (list or list of lists):
        List of enriched features per cluster. 

    """

    # Get the feature_names
    if feature_names == None and type(data) == pd.DataFrame:
        feature_names = data.columns.to_list()

    elif feature_names == None and type(data) == np.ndarray:
        print("No feature names provided, feature enrichment will not be meaninful.")

    try:
        nmf = NMF(n_components=n_clusters).fit(data)

    except:
        nmf = sklearn.decomposition.NMF(n_components=n_clusters).fit(data)

    # Get W and H matrices
    nmf_W = nmf.transform(data)
    nmf_H = nmf.components_

    # Initialize list to store cluster labels
    cluster_labels = []

    # Store the number of samples (rows in the original dataset)
    n_samples = nmf_W.shape[0]

    #  Get sample cluster labels iterating over W matrix's rows
    for i in range(n_samples):
        # Get the index for which the value is maximal.
        cluster_labels.append(nmf_W[i].argmax())

    # Initialize to store enriched features
    enriched_feats = []

    # Get features cluster coefficients
    # iterates over the H rows
    for cluster_idx, cluster in enumerate(nmf_H):

        top_feats = [
            marker_dict[feature_names[i]] for i in cluster.argsort()[: -n_enriched_feats - 1 : -1]
        ]

        enriched_feats.append(top_feats)
        
    if n_enriched_feats == 1: 
        enriched_feats = np.array(enriched_feats).flatten()

    return nmf_W, nmf_H, cluster_labels, enriched_feats



def make_hist_from_latent(data, sample_id, cols_list, channels,
                          resolution, export_path, export = True, **kwargs):

    """
    Returns an n-d histogram from the latent space representation
    of a sample. 

    Params 
    ------

    data (pd.DataFrame)
        Dataframe containing latent representation coordinates 
        of a sample. 

    sample_id (str)
        Name of the sample to encode. 

    cols_list (list)
        List of the columns to extract the representation from. 
        
    channels (int)
        Size of the first dimension which will serve as a "channel"
        dimension.
    
    resolution (int)
        Size of the second and third dimension. 

    export (bool, default = True )
        Whether to export the resulting histogram as an image.
    
    **kwargs
        Keyword arguments passed to the np.histogramdd function. 
    
    Returns 
    -------
    hist(array-like)
        3D histogram of the sample's latent space representation. 
    
    """
    
    # Extract latent space to compute bounds 
    data_ = data[cols_list]
    # Round to first decimal 
    mins_ = np.round(data_.min() - 0.01, 1)
    maxs_ = np.round(data_.max() - 0.01, 1)

    print(f'maxs: {maxs_}, mins: {mins_}')

    bounds = list(zip(mins_, maxs_))
    

    # Get the data for the corresponding 
    samples = data[data['sample_id_unique'] == sample_id][cols_list].values

    print(f'Number of cells in dataset: {samples.shape}' )

    # Get n-d hist of data 
    hist, edges = np.histogramdd(
        samples,
        bins = (channels, resolution, resolution), 
        range = bounds,
        #normed = True, 
        **kwargs
    )

    if export: 
        
        sample_id = sample_id.split(' ')[0].split('/')[0]
        fname = export_path + 'latent_' + sample_id + '.npy'
        np.save(fname, hist)
        
        return fname, hist
    
    else:    
        return hist 





def simple_bootstrap_df(df, n_boostrap = 10, reset_index = False): 
	"""
	
	"""
	n_cells = df.shape[0]

	# Sample n_cells 
	#df_list =  

	df_bootstrap = df.copy()

	# Define original data as bs_sample zero
	df_bootstrap['bs_sample'] = 0

	for i in range(1, n_boostrap +1):
		bootstrap_sample = df.sample(df.shape[0], replace = True)
		bootstrap_sample['bootstrap_sample'] = i

		df_bootstrap = pd.concat([df_bootstrap, bootstrap_sample])

	#
	if reset_index: 
		df_bootstrap = df_bootstrap.reset_index(drop = True)

	return df_bootstrap



def stratified_bootstrap_df(df, col_stratify = 'cell_ontology_class',
 	n_bootstrap = 10, reset_index = False, verbose = True):
	"""
	Returns a new dataframe by sampling n_bootstrap datasets with replacement from 
	the original dataframe df. In this version it also keeps the relative proportion of
	datapoints on col_stratify. 
	

	Params 
	-------
	df ()

	col_stratify (str, default = 'cell_ontology_class')

	n_bootstrap(int, defualt = 10)
		Number of bootstrap samples. 

	reset_index(bool, default = True)
		Whether to reset the index of the new dataset. 

	Returns 
	-------
	df_bootstrap(pd.DataFrame)
		Dataset with bootstrapped samples. It contains the original dataset. 

	"""

	df_bootstrap = df.copy()

	# Name the original dataset bs sample zero
	df_bootstrap['bootstrap_sample'] = 0

	for i in range(1, n_bootstrap + 1): 

		sampling_ix = (
		    df.groupby(col_stratify)
		    .apply(lambda group_df: group_df.sample(group_df.shape[0], replace=True))
		    .index.get_level_values(1)
		)

		bootstrap_sample = df.loc[sampling_ix, :]

		bootstrap_sample['bootstrap_sample'] = i

		df_bootstrap = pd.concat([df_bootstrap, bootstrap_sample])

	if reset_index: 
		df_bootstrap = df_bootstrap.reset_index(drop = True)

	if verbose: 
		n_non_sampled = len(set(df.index) - set(df_bootstrap[df_bootstrap['bootstrap_sample'] != 0].index))

		print(f'Number of indices not sampled: {n_non_sampled}')

	return df_bootstrap




def get_cell_type_props(df, var_id, agg_var = 'age'):

	"""
	Returns a dataframe with cell type proportions across the Tabula Muris' dataset
	tissues, for a specified age section. 

	Params
	------
	df (pd.DataFrame)
		Summary dataframe of Tabula Muris senis. It should contains the following 
		columns = ['age', 'age_code', 'tissue', 'cell_ontology_class']

	agg_var (str)
		Particular variable to select for. 
		Common used examples are age and mouse.id.

	Returns 
	-------

	normalized_tissue(pd.DataFrame)
		Tissue with cell type proportions 
	"""
	sub_df = df[df[agg_var] == var_id]

	tissue_counts = (
	    sub_df.groupby(["tissue", "cell_ontology_class"])
	    .size()
	    .to_frame()
	    .reset_index()
	    .rename(columns={0: "counts"})
	)

	pivot = pd.pivot_table(
	    data = tissue_counts,
	    index ="tissue",
	    columns = 'cell_ontology_class',
	    fill_value = 0
	)

	tissue_total = pivot.sum(axis = 1)

	normalized_tissue = pivot.div(tissue_total, axis = 'rows')
	normalized_tissue = normalized_tissue.droplevel([None], axis = 1)

	# Annotate with age column
	normalized_tissue =normalized_tissue.T
	normalized_tissue[agg_var] = var_id

	normalized_tissue = normalized_tissue.fillna(0)

	return normalized_tissue


def get_single_tissue(df_tissues, tissue_name, agg_var, drop_zeros = True):

	"""
	Returns cell type proportions of a single tissue. 
	
	Params 
	------
	df_tissues(pd.DataFrame)
		Pivoted dataframe of (cell_types, tissues) shape.
	
	tissue_name (str)
		Name of the tissue to select 

	agg_var (str)
		Particular variable to select for.
		Common used examples are age and mouse.id.

	Returns 
	-------
	tissue (pd.DataFrame)
		Subdataframe of a single tissue of shape 

	"""

	tissue = pd.pivot_table(
			data = df_tissues[[tissue_name, agg_var]].reset_index(), 
			index = 'cell_ontology_class',
			columns = agg_var,
			aggfunc= 'mean',
			fill_value = 0
		)
	# Drop the multiindex name that corresponds to the tissue name 
	tissue = tissue.droplevel(None, axis = 1)

	
	if drop_zeros: 
		# Eliminate all cell types with no counts in tissue 
		tissue = tissue[tissue.sum(axis = 1) > 0]
		# Drop the individuals with no counts for given tissue
		tissue = tissue.loc[:, (tissue !=0).any(axis = 0)]

	# Renormalize due to aggregation effects
	if agg_var == 'age': 
		tissue = tissue / tissue.sum(axis = 0)

	return tissue


#def make_data_array(): 

