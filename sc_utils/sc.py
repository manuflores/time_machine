import scipy.io as sio
import scipy.stats as st
#import scanpy as sc
import numpy as np
#import numba
import pandas as pd 
import anndata as ad
from scipy import sparse 
#from psutil import virtual_memory
import time 
import os
import seaborn as sns
import toolz as tz
import h5py
import tqdm 
from sklearn.utils import sparsefuncs
# Correct p-vals using Benjamini-Hochberg procedure
from statsmodels.stats.multitest import multipletests

def check_sparsity(adata): 
	"Returns the percentage of zeros in dataset."

	if not sparse.isspmatrix_csr(adata.X):
		adata.X = sparse.csr_matrix(adata.X)

	sparsity = (1 - adata.X.data.shape[0] / (adata.n_obs*adata.n_vars))*100

	return sparsity 


# Make curried to allow kwarg calls on tz.pipe()
@tz.curry
def get_count_stats(
	adata,
	mt_prefix = None,
	ribo_prefix = None)-> ad.AnnData:

	"""
	Returns an AnnData with extra columns in its `obs` object 
	for the number of counts per cell `n_counts` (and log10 (counts) ),
	abd the number of expressed genes in each cell `n_genes`.
	Additionally it can get the fraction of mitochondrial and ribosomal
	genes if prefixes are provided.
	
	TODO: Add filtering functionality
	
	Params
	------
	adata (ad.AnnData)
		Input dataset in AnnData format. It should contain a count matrix
		(cells x genes) as the `.X` object in the AnnData.

	mt_prefix (str, default = 'MT-'): 
		Prefix to match mitochondrial genes.
		For human the prefix is `MT-` and for the mouse is `mt-`. 

	ribo_prefix(default=None)
		For human the prefixes are ('RPS', 'RPL').

	Returns
	-------
	adata (ad.AnnData)
		AnnData with columns in the `.obs` dataframe corresponding to
		count stats.
	"""

	if not sparse.isspmatrix_csr(adata.X): 
		adata.X = sparse.csr_matrix(adata.X)

	# Number of transcripts per cell
	adata.obs['n_counts'] = adata.X.sum(axis = 1)
	adata.obs['log_counts'] = np.log10(adata.obs.n_counts)

	# Number of genes with more than one count
	adata.obs['n_genes'] = (adata.X > 0).sum(axis = 1)

	# Get mitochondrial and ribosomal genes 
	if mt_prefix is not None: 
		# Use string methods from pandas to make bool array
		mito_genes = adata.var.gene_name.str.startswith(mt_prefix)

		if mito_genes.sum()> 1:

			# Compute the fraction of mitochondrial genes
			adata.obs["frac_mito"] = adata[:, mito_genes].X.A.sum(axis =1) / adata.obs.n_counts

	if ribo_prefix is not None: 
		
		if isinstance(ribo_prefix, (list, tuple)): 
			# Initialize bool array 
			ribo_genes = np.zeros(adata.n_vars, dtype = bool)

			# Loop through each prefix and flip to True 
			# where we get a match. 
			for prefix in ribo_prefix:
				ribo_genes_tmp = adata.var.gene_name.str.startswith(prefix)
				ribo_genes +=ribo_genes_tmp

			if ribo_genes.sum()> 1:
				adata.obs["frac_ribo"] = adata[:, ribo_genes].X.A.sum(axis =1) / adata.obs.n_counts

	return adata 


# Curry to be able to add arguments in a tz.pipe
@tz.curry
def lognorm_cells(
	adata_,
	scaling_factor = 1e4,
	log = True)-> ad.AnnData: 

	"""
	Cell count normalization as in scanpy.pp.normalize_total.
	Expects count matrix in sparse.csr_matrix format. 

	Each gene's expression value in a given cell is given by : 

	g_i = \mathrm{ln} ( \frac{g_i \times \beta }{\sum g_i} + 1 )
	
	where β is the scaling factor. 

	Params
	------
	adata_ (ad.AnnData): 
		Count matrix with cell and gene annotations. 
	
	scaling_factor(float, default = 1e4)
		Factor to scale gene counts to represent the counts in 
		the cell. If scaling_factor =1e6, the values will 
		represent counts per million. 
	
	log (bool, default = True)
		Optional argument to allow for returning the scaled cells 
		without normalizing. 
	
	Returns 
	-------
	adata (ad.AnnData): 
		Anndata with normalized and log transformed count matrix. 
	"""
	
	# Make a copy because normalization is done in-place
	adata = adata_.copy()

	if not sparse.isspmatrix_csr(adata.X):
		adata.X = sparse.csr_matrix(adata.X)

	# Get total counts per cell from `obs` df
	if 'n_counts' in adata.obs.columns:
		counts = adata.obs.n_counts.values

	else: 
		counts = adata.X.sum(axis = 1).flatten()

	# Convert to numpy matrix to array to be able to flatten
	scaled_counts = np.array(counts).flatten() / scaling_factor

	# Efficient normalization in-place for sparse matrix
	sparsefuncs.inplace_csr_row_scale(adata.X, 1/scaled_counts)

	# Call the log1p() method on the csr_matrix 
	if log:
		
		adata.X = adata.X.log1p()

	return adata

# Curry to enable adding arguments in a tz.pipe()
@tz.curry
def cv_filter(
	adata,
	min_mean = 0.025,
	min_cv= 1,
	return_highly_variable = False)-> ad.AnnData:

	"""
	Performs the Coefficient of Variation filtering according 
	to the Poisson / Binomial counting statistics. The model assumes 
	the coefficient of variation per gene is given by : 

	\mathrm{log} (CV) \approx - \frac{1}{2}\mathrm{log} (\mu) + \epsilon
 

	The values will be computed assuming a normalized and 
	log-scaled count matrix. 
	
	Params 
	------
	min_mean (float, default = 0.025). 
		Lower bound cutoff for the mean of the gene feature.
	
	min_cv (float, default = None)
		Lower bound for the coefficient of variation of the 
		gene feature. Recommended value 1. 

	return_highly_variable(bool, default = True)
		Whether to return an AnnData with the columns corresponding
		to only the highly variable genes. 
		Note: even when running with `return_highly_variable=False`
		the function will return genes only with nonzero mean and 
		nonzero variance, i.e. it will discard those genes.
	
	Returns 
	-------
	adata_filt (ad.AnnData)
		AnnData with coeffifient of variation stats on the `var`
		dataframe.
	"""

	# Calculate mean and variance across cells 
	mean, var = sparsefuncs.mean_variance_axis(adata.X, axis = 0)

	# Check if there are nonzero values for the mean or variance 
	ix_nonzero = list(set(np.nonzero(mean)[0]).intersection(set(np.nonzero(var)[0])))

	if len(ix_nonzero) > 0: 
		# Use numpy-like filtering to select only genes with nonzero entries
		adata = adata[:, ix_nonzero].copy()
		
		# Recompute mean and variance of genes across cells
		mean, var = sparsefuncs.mean_variance_axis(adata.X, axis = 0)
		
		# Get nonzero mean indices
		nz = np.nonzero(mean)

		# Check that there are no nonzero mean values
		assert adata.n_vars == nz[0].shape[0]


	std_dev = np.sqrt(var)

	# Element-wise coefficient of variation
	cv = std_dev / mean
	log_cv = np.log(cv)
	log_mean = np.log(mean)

	df_gene_stats = pd.DataFrame(
	    np.vstack([mean, log_mean, var, cv, log_cv]).T,
	    columns=["mean", "log_mean", "var", "cv", "log_cv"],
	    index = adata.var.index
	)

	new_adata_var = pd.concat(
	    [adata.var, df_gene_stats], 
	    axis = 1
	)

	adata.var = new_adata_var

	slope, intercept, r, pval, stderr = st.linregress(log_mean, log_cv)

	# Check that slope is approx -1/2
	print(f'The slope of the model is {np.round(slope,3)}.')

	poisson_prediction_cv = slope*log_mean + intercept

	# Binary array of highly variable genes 
	gene_sel = log_cv > poisson_prediction_cv
	
	adata.var['highly_variable'] = gene_sel.astype(int)

	if min_mean and min_cv is not None: 
		adata_filt = adata[:,((adata.var.highly_variable == True)&\
								(adata.var['mean'] > min_mean)&\
								(adata.var['cv'] > min_cv))].copy()
	else: 
		adata_filt = adata[:, adata.var.highly_variable == True].copy()
	
	if return_highly_variable: 
		return adata_filt

	else: 
		return adata


def nmf_wrapper_sc(
    data:np.array,
    marker_dict,
    n_clusters = 10,
    n_enriched_feats=1,
    feature_names=None, 
    )->tuple:
    
    """
    Python wrapper to implement NMF algorithm. 

    It returns the W and H matrices, the cluster_labels per sample, 
    and the top enriched features per cluster. 

    Assumes that either that either sklearn or the 
    sklearn.decomposition.NMF class are loaded in 
    the workspace.

    Params
    -------
    data: (pd.DataFrame or np.array)
        Dataset to be decomposed. All datapoints need to
        be non-negative.

    marker_dict(dict)

    feature_names (array-like): 
        Name of the columns of the dataset.
        This is used to compute the feature enrichment.

    n_enriched_feats (int): 
        number of top enriched features to extract from
        each cluster. In the scRNAseq case, it amounts to get 
        the top genes in each cluster.

    n_clusters (int): 
        Number of components to make the matrix factorization.

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
    from sklearn.decomposition import NMF

    # Get the feature_names
    if feature_names == None and type(data) == pd.DataFrame:
        feature_names = data.columns.to_list()

    elif feature_names == None and type(data) == np.ndarray:
        print("No feature names provided, feature enrichment will not be annotated.")
    else:
    	pass

	#nmf = NMF(n_components=n_clusters).fit(data)
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


def safe_gene_selection(
    adata,
    input_list, 
    gene_colname = 'gene_name', 
    #keep_order=False
    )-> ad.AnnData:
    """
    Returns a new adata with a subset of query genes. 

    TODO: Handle case when we want to keep the query list. 

    Note: It will only return the genes that are in the dataset. 
    If any of the query genes are not in the dataset, the gene names
    will be dismissed. If you're not too sure of the exact gene names
    check the `df.gene_colname.str.contains()` or the 
    `df.gene_colname.str.startswith()` function. 

    Params
    ------
    adata(ad.AnnData)
        Dataset to select from. 

    input_list(array-like)
        Query list with gene names. 

    gene_colname (str, default = 'gene_name')
        Name of the column in the .var object from which to 
        make the query against. 

    Returns
    -------
    new_adata (ad.AnnData)
        Subset of original anndata containg only query genes. 

    Example
    -------

    # Initalize dummy list and shuffle it
    gene_names = list('ABCDEFGHIJ')
    rng = np.random.default_rng(seed = 9836)
    gene_names = rng.permutation(gene_names)
    print(gene_names)

    # Create adata with random 5 cell 10 gene count matrix
    a = ad.AnnData(
        X = np.random.random((5, 10)),
        var= pd.DataFrame(gene_names, columns = ['gene_name'])
    )

    my_list = ['A', 'C', 'B','D']

    ada_new = sc.safe_gene_selection(a, my_list)

    print(ada.var.gene_name.values)
    >>> array(['A', 'B', 'C', 'D'], dtype=object)

    """

    # Gets the indices of the rows contained in my_list using bool array
    isin_indexer = adata.var[gene_colname].isin(input_list).values.nonzero()[0]
    
    # Returns the indices that sort the values 
    # selected with the indexer array
    new_ixs = np.argsort(adata.var[gene_colname].values[isin_indexer])
    
    isin_list_sorted_ixs = isin_indexer[new_ixs]
    
    adata_new = adata[:, isin_list_sorted_ixs].copy()
    
    return adata_new



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
	TO-DO: write documentation. 
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

	# Try alternative 
	#pivot = sub_df.groupby(["tissue", "cell_ontology_class"]).size().unstack()

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

def read_cancer_adata(
	path, 
	h5_fname, 
	meta_fname)->(ad.AnnData, pd.DataFrame):
	
	"""
	Load and convert .h5 file to AnnData from the publication:

	https://www.nature.com/articles/s41591-020-0844-1

	"""

	# Load metadata file 
	df_meta = pd.read_csv(path + meta_fname)

	h5_file = h5py.File(path + h5_fname)

	dset_name = list(h5_file.keys())[0]

	data = h5_file[dset_name]["data"][()]
	indices = h5_file[dset_name]["indices"][()]
	indptr = h5_file[dset_name]["indptr"][()]
	shape = h5_file[dset_name]["shape"][()]
	gene_names = h5_file[dset_name]["gene_names"][()]
	gene_names = [x.decode('ascii') for x in gene_names]

	barcodes = h5_file[dset_name]["barcodes"][()]
	barcodes = [x.decode('ascii') for x in barcodes]

	adata = ad.AnnData(
		X = sparse.csc_matrix(
				(data, indices, indptr),
				shape = shape
			).T,
		#obs = df_meta, 
		var= pd.DataFrame(gene_names, columns = ['gene_names'])
		)

	# Check if count matrix is in csr format
	#assert sparse.isspmatrix_csr(adata.X)

	return adata, df_meta 




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



def confusion_matrix(pred_labels, true_labels): 
    """
    Returns a confusion matrix from a multiclass classification 
    set of labels. 
    
    Params 
    ------
    pred_labels (array-like):
        List of labels as predicted by a classification algorithm. 
    
    true_labels (array-like): 
        List of ground truth labels. 
    
    Returns 
    -------
    conf_mat (array-like): 
        Confusion matrix. 
    """
    
    n_labels = int(max(np.max(pred_labels), np.max(true_labels)) + 1)
    conf_mat = np.zeros(shape = (n_labels, n_labels))
    
    for (i, j) in zip(pred_labels, true_labels): 
        conf_mat[i,j] +=1
        
    return conf_mat

def element_wise_entropy(px):
    """
    Returns a numpy array with element wise entropy calculated as -pi*log_2(p_i).
    
    Params
    ------
    px (np.array)
        Array of individual probabilities, i.e. a probability vector or distribution.
    
    Returns
    -------
    entropy (np.array)
        Array of element-wise entropies.
    """
    if isinstance(px, list):
        px = np.array(px)
        
    # Make a copy of input array
    entropy = px.copy()
    
    # Get indices of nonzero probability values
    nz = np.nonzero(entropy)
    
    # Compute -pi*log_2(p_i) element-wise
    entropy[nz] *= - np.log2(entropy[nz])
    
    return entropy


def entropy(ps): 
    "Returns the entropy of a probability distribution `ps`."
    # Get nonzero indices
    nz = np.nonzero(ps)

    # Compute entropy for nonzero indices
    entropy = np.sum(-ps[nz]*np.log2(ps[nz]))
    
    return entropy



# EXPERIMENTAL: gene analysis

def score_cells_percentile(val, percs, score_hi=1, score_low=-1): 
    """
    Returns a score given value of a single cell.
    
    Cells with score = -1  will be cells in low prototype (cells below the 
    bottom percentile). Conversely, cells with a score =1, will be
    the top prototype. 
    
    Params
    ------
    val (float): 
        Value of the cell at given dimension. 
    percs (tuple): 
        Tuple of top and bottom percentiles to compare with.
    
    Returns 
    -------
    score(int)
        Score of the cell.
    """
    low_perc, hi_perc = percs
    if val > hi_perc:
        score = score_hi
    elif val < low_perc: 
        score = score_low
    else: 
        score = 0
    return score

def score_cells_along_component(df_proj, latent_dim, percs = (10, 90)): 
    """
    Returns an array of scores for each of the cells along 
    a latent dimension. 
    
    Params 
    ------
    latent_dim (array-like)
        Array of values for each cell in the latent dimension or component. 
    percs (tuple, default= (10,90))
        Low and bottom percentiles to compute. 
    
    Returns
    -------
    scores (array-like)
        List of scores for each cell. 
    """
    low_perc, hi_perc = np.percentile(df_proj[latent_dim], percs)
    
    scores = [score_cells_percentile(p, (low_perc, hi_perc)) for p in df_proj[latent_dim]]
    
    return scores



# def get_scanpy_deg_report_df(adata, groups = ('-1','1')):
#     """
#     Returns a report dataframe of differentially expressed genes. 
#     It expects an adata with a report dictionary from the output of
#     scanpy.tl.rank_genes_groups(). 
    
#     Params 
#     ------
#     adata (ad.AnnData)
#         AnnData with rank_genes_groups dictionary in `.uns` object.
#         Ideally, this adata would only contain "prototype" cells,
#         that is, the cells on the extremes of a given component. 
        
#     groups (tuple, default = (-1,1))
#         Tuple of groups for which to extract the DEG results.
    
#     """
    
#     deg_result_dict = adata.uns['rank_genes_groups']
    
#     # Make a report dataframe. 
#     df_report = pd.DataFrame()

#     for g in groups:
#         df = pd.DataFrame(
#             np.vstack(
#                 [deg_result_dict["names"][g],
#                  deg_result_dict["logfoldchanges"][g],
#                  deg_result_dict["pvals_adj"][g]]
#             ).T,
#             columns=["gene_name", "log_fc", "pval_adj"],
#         )

#         df["group"] = g

#         df_report = pd.concat([df_report, df])
    
#     return df_report


# Curry to enable passing arguments to .apply() method
@tz.curry
def is_significant(l1_score, thresh_over = -.8, thresh_under = .5):
    if l1_score < thresh_over or l1_score > thresh_under: 
        return 1
    else:
        return 0

def over_under_exp(l1_score):
    "To be used after extracting significant genes."
    if l1_score < 0 : 
        return 'over'
    elif l1_score > 0:
        return 'under'
    else:
        return None


def deg_analysis(
	adata,
	groupby_col = 'scores', 
	groups = [-1,1], 
	thresh_under= .5, 
	thresh_over = -.5,
	return_all_genes = False
	)->pd.DataFrame:
	"""
	TO-DO: Enable parallelism 

	Returns a dataframe containing the report of differential gene expression (DEG) analysis 
	between two groups of cells. It tests the hypothesis that all genes in group 1 
	have higher values (are overexpressed) than genes in group -1 using non-parametric tests. 
	
	It assumes genes are log-normed and expects that the AnnData has a column that indicates 
	the groups of cells to make the analysis on, in the `obs` DataFrame. 

	The function uses the Kolmogorov-Smirnov (KS) 2-sample test on the ECDFs and the L1-norm
	on the histogram of the gene distributions, respectively. We use the L1 norm to assign a
	significance of over or underexpression. We consider a L1 score < -0.5 for a gene to be 
	overexpressed in group 1 and a L1 score of > 0.5 to be underexpressed in group 1.
	
	Params 
	------
	adata (ad.AnnData)
		AnnData with an .obs dataframe containing a column to select the groups.
	
	group_by (str, default = 'scores')
		Column that indicates the groups to make the test on. 

	groups (list, default = (-1,1))
		Name of the groups. 
	
	Returns
	-------
	df_report (pd.DataFrame)


	Notes
	-----
	The KS runs the following hypothesis test: 

	* *H0*: The ECDF of gene$_i$ in (group 1) $F$ is equal
	to the distribution of (group -1) $G$, i.e. $F(u) = G(u)$ for all quantiles $u$. 
	
	* *H1* : The distribution $F$ for gene$_i$ has **higher values** in (group 1) compared
	to (group -1). Having higher values implies that for a given quantile *u*, 
	the values of the CDF of (group 1) will be lower than those of the non-lactating state,
	that is $F(u) <= G(u)$ for all quantiles $u$. 
	
	The L1 norm is computed as reported in [Chen *et al* (2020)]
	(https://www.pnas.org/content/117/46/28784) 

	"""
	df_report = pd.DataFrame()

	# Get gene name from the `.var` dataframe
	gene_names = adata.var.gene_name.values


	# Unpack groups
	gp_lo, gp_hi = groups 

	# Loop through all genes 
	for ix in tqdm.tqdm(range(adata.n_vars)):
	    
	    gene_name = gene_names[ix]
	    
	    # Extract distributions as arrays 
	    gene_distro_lo = adata[adata.obs[groupby_col]== gp_lo, ix].X.A.flatten()
	    gene_distro_hi = adata[adata.obs[groupby_col]== gp_hi, ix].X.A.flatten()
	    
	    # Run KS test 
	    # Second element in results tuple is P-value
	    ks_test_pval = st.ks_2samp(
	        gene_distro_lo,
	        gene_distro_hi,
	        alternative = 'greater',
	        #mode = 'exact'
	    )[1]
	    
	    # Compute L1 norm
	    l1_score = l1_norm(gene_distro_hi, gene_distro_lo)
	    
	    # log2 fold-change of means 
	    log_fold_change = np.log2(gene_distro_hi.mean() / gene_distro_lo.mean())
	    
	    # Wrap into a dataframe 
	    #pd.DataFrame(
	    df = {
            'gene_name': gene_name,
            'ks_pval' : ks_test_pval,
            'l1_score' : l1_score,
            'log_fc': log_fold_change
	        }
	    #)
	    
	    df_report = df_report.append(df, ignore_index = True)

	df_report.reset_index(drop =True, inplace = True)

	df_report = df_report.sort_values(by = ['l1_score'], ascending = True)

	_, pvals_corr, _, _ = multipletests(df_report.ks_pval.values, method = 'fdr_bh')
	df_report['ks_pval_BH'] = pvals_corr


	if return_all_genes:
		return df_report 
	else:
		# 
		df_report['is_signif'] = df_report.l1_score.apply(
				is_significant(thresh_over = thresh_over, thresh_under = thresh_under)
			)

		df_deg = df_report[df_report['is_signif']==1]

		df_deg['deg_type'] = df_deg.l1_score.apply(over_under_exp)

		return df_deg


def deg_test(
	adata, annot_cols, groupby = 'scores', groups = [-1,1], return_melted = True
	)->(pd.DataFrame): 

	"""
	Runs DEG analysis and return dataframes for visualization. 
	
	Params
	------
	adata (ad.AnnData)
		Annotated count matrix in AnnData format. 

	annot_cols (list)
		Subset of the columns in the adata `obs` dataframe
		to use for visualization. 

	groupby (str, default = 'scores')
		Column that indicates the groups to make the test on. 
	
	groups(list, default = [-1,1])
		Name of the groups.

	return_melted(bool, default = True)
		If set to True, it returns a melted version of the df_viz
		dataframe, useful for plotting distribution plots (e.g. boxplots,
		violinplots) of the differentially expressed genes.
		
	
	Returns 
	-------
	df_deg (pd.DataFrame)
		Report dataframe from DEG analysis. 

	df_viz (pd.DataFrame)
		DataFrame to visualize scatterplots colored by gene counts,
		or heatmaps. 

	df_distro_viz (pd.DataFrame, optional)
		Tidy dataframe to visualize distributions of genes. 

	Example
	-------
	### Assuming an anndata is loaded in workspace

	# Run DEG analysis and plot overexpressed gene distributions. 
	df_deg, df_viz, df_distro_viz = sc.deg_test(adata, annot_cols)
	
	df_viz_prototype =  df_distro_viz[df_distro_viz['scores'] != 0]

	plt.figure(figsize = (4, 9))
	sns.violinplot(
	    data = df_viz_prototype[df_viz_prototype['deg_type'] == 'over'],
	    y = 'gene_name',
	    x = 'log(counts)', 
	    hue = 'classification column', 
	    palette= viz.get_binary_palettes()[1],
	    split = True,
	    scale = 'width', 
	    inner = 'quartile',
	    cut = 0
	)
	
	"""

	df_deg = deg_analysis(adata, groupby_col = groupby, groups = groups)

	overexp_genes = df_deg[df_deg['deg_type'] == 'over'].gene.to_list()
	underexp_genes = df_deg[df_deg['deg_type'] == 'under'].gene.to_list()

	deg_gene_list = overexp_genes + underexp_genes

	# Get a subset of the original adata containing the DE genes only. 
	de_genes_adata = safe_gene_selection(adata, deg_gene_list)

	# Make DataFrame for visualization 
	de_genes_df = pd.DataFrame(
		de_genes_adata.X.A,
		columns = de_genes_adata.var.gene_name
	)

	# Concatenate annotation with gene values
	# This dataset serves to make scatterplots using
	# the gene expression values to color the dots
	df_viz = pd.concat(
	    [adata.obs[annot_cols],
	     de_genes_df.set_index(adata.obs.index)],
	    axis = 1
	)

	if return_melted:  
		# Melt to visualize the distribution of genes 
		# across groups, e.g. violinplots 
		df_distro_viz = pd.melt(
		    df_viz,
		    id_vars = annot_cols,
		    value_vars = de_genes_df.columns.to_list(),
		    var_name = 'gene_name',
		    value_name = 'log(counts)'
		)

		# Check that melt frame is of length n_de_genes * n_cells
		assert df_distro_viz.shape[0] == len(deg_gene_list)*adata.n_obs

		# Add column indicating whether gene is over or underexpressed
		# in group 1
		gene_dge_type_mapper= dict(df_deg[['gene_name', 'deg_type']].values)
		df_distro_viz['deg_type'] = df_distro_viz.gene_name.map(
			gene_dge_type_mapper
		)


		return df_deg, df_viz, df_distro_viz

	else: 
		return df_deg, df_viz

def annot_genes(gene, gene_sets_tuple):
	"""
	Helper function to annotate a given gene in prototype cells
	enriched genes.
	Returns -1 if gene is in bottom prototype, 1 if in top prototype. 
	"""

	gene_set_low, gene_set_hi = gene_sets_tuple
    
	if gene in gene_set_low:
	    y = -1
	elif gene in gene_set_hi: 
	    y = 1
	else: 
	    y = 0

	return y

def fisher_enrichment_test(
	df_annot,
	annotation,
	group, 
	group_colname = 'cluster_labels', 
	n_top = 5
	)->pd.DataFrame: 
    """
    Returns a report dataframe with the top 5 enriched functions
    for a given subset of data. This function is especially suited
    for statistical enrichment tests after clustering. 
    
    Params 
    ------
    df_annot (pd.DataFrame)
        Annotated dataframe containing the 'annotation' column 
        and a 'clus_col_name' column. 
    
    annotation (str)
        Annotation to make the enrichment test on. In the case 
        of gene set enrichment this could be a Gene Ontology 
        or COG annotation. 
    
    cluster (int or str)
        Cluster (or in general group of data points) to test.
    
    clus_col_name (str)
        Name of the cluster column in the df_annot dataframe.
    
    Returns 
    -------
    df_report (pd.DataFrame)
        Report dataframe with pvalues and annotation names. 
    
    """
    # Get subset of completely annotated genes 
    df_test = df_annot[pd.notnull(df_annot[annotation])]

    # Number of genes with valid annotation 
    M = df_test.shape[0]

    # Extract data for given cluster
    df_clus = df_test[df_test[group_colname] == group]

    # Get n_top categories to test (defaults to 5)
    cats = df_clus[annotation].value_counts().head(n_top).index.to_list()
    
    # Number of genes in the cluster (sample size)
    N = df_clus.shape[0]
    
    # Initialize pvalue array 
    pvals = np.empty(len(cats))
    
    # Loop through the top categories
    for i, cat in enumerate(cats): 
        
        df_cat = df_test[df_test[annotation] == cat]
        
        # Total number of genes that map to given category (total number of white balls)
        n = df_cat.shape[0]
        
        # Number of genes inside cluster that map to given category (number of white balls in sample)
        x = df_clus[df_clus[annotation] == cat].shape[0]
        
        # Sweep through the probabilities from x to n 
        pmfs = st.hypergeom.pmf(k = np.arange(x, n + 1), N = N, n = n, M = M)
        
        # Compute pval
        pvals[i] = pmfs.sum()
    
    # Save results
    df_report = pd.DataFrame(
        {'categories': cats, 'pval': pvals}
    )

    df_report['group'] = group
    df_report['annot'] = annotation

    return df_report


def run_fisher_test_go(
	df_report_deg,
	path,
	groupby, 
	clus_col_name = 'prototype', 
	groups = [-1,1],
	n_top = 5,
	organism = 'human'
	)->(pd.DataFrame, pd.DataFrame):
    """
    Returns (1) a dataframe with enriched Gene Ontology terms 
    for a set of differentially expressed genes, and 
    (2) a more detailed dataframe of the Gene Ontology terms with their GO ID,
    and whether or not they are Transcription Factors.

    Params
    ------
    df_report_deg(pd.DataFrame)
        Differentially DEG report dataframe. 
    
    path_to_ds(str)
        Path to drug screen dataset. 

    n_top (int)
        Number of categories to compute the Fisher enrichment test. 
    
    organism (str, default = 'human')
        Name of the organism. Currently handling ['human', 'mouse'].
        
    Returns 
    ------- 
    df_enrichment_report(pd.DataFrame)
        Report of enriched GO terms with corresponding pvals.
    
    df_go_red (pd.DataFrame)
        Dataframe of enriched genes with their corresponding 
        GO biological process. 
    """

    # path = 'path_to_drug_screen'


    # Unpack groups
    gp_lo, gp_hi = groups
    
    #Load Gene Ontology dataset (from QuickGO annotation)
    df_go =  pd.read_csv(path + 'go/go_lite_' + organism +'.csv')

    # Get gene names of DEG genes
    genes_disease = df_report_deg[df_report_deg[groupby]== gp_lo]['gene_name'].values
    genes_healthy = df_report_deg[df_report_deg[groupby] == gp_hi]['gene_name'].values
    
    gene_sets = (genes_disease, genes_healthy)
    
    # Annotate genes as prototypes:
    # These annotations will help the Fisher enrichment test function
    df_go['deg_type'] = [annot_genes(g, gene_sets) for g in df_go.gene_name.values]

    # Compute FET for bottom and top prototypes
    df_enrichment_lo_prot = fisher_enrichment_test(
        df_annot = df_go, 
        group_colname = clus_col_name,
        annotation = 'GO NAME', 
        group = -1,
        n_top = n_top
    )

    df_enrichment_top_prot = fisher_enrichment_test(
        df_annot = df_go, 
        group_colname = clus_col_name,
        annotation = 'GO NAME', 
        group = 1, 
        n_top = n_top
    )
    
    df_enrichment_report = pd.concat([df_enrichment_lo_prot,
                                     df_enrichment_top_prot])
    
    df_enrichment_report = df_enrichment_report.\
                        sort_values(by = ['group', 'pval']).reset_index(drop = True)
    
    #df_enrichment_report = df_enrichment_report.rename(columns = {'clusters':'prototype'})

    # Get GO categories for differentially expressed genes
    # i.e. in prototypes -1 or 1 (not in 0).
    try:
        df_go_red = df_go[df_go['deg_type'].isin([-1,1])]
    except: 
        df_go_red = df_go[df_go['deg_type'].isin(['-1','1'])]
        
    # Load Transcription Factor gene names
    path_to_tfs = path + '../trns/' + organism + '_trn/'+ organism + '_tfs.csv'
    
    tf = pd.read_csv(path_to_tfs)
    
    # Annotate if DEG genes are TFs
    df_go_red['is_tf'] = df_go_red.gene_name.apply(
        lambda x: 1 if x in tf['gene_name'].unique() else 0
    )
    
    return df_enrichment_report, df_go_red


def ecdf(x, plot = False, label = None)->(np.array, np.array):
    '''
    Returns ECDF of a 1-D array. Optionally  

    Params
    ------

    x(array or list)
    	Input array, distribution of a random variable.
    
    plot (bool, default= False)
    	If True return the plot of the ECDF

    label(str)
    	Label for the plot
    
    Returns 
	-------
    x_sorted : sorted x array.
    ecdf : array containing the ECDF of x.

    '''
    x_sorted = np.sort(x)
    
    n = len (x)
    

    ecdf = np.linspace(0, 1, len(x_sorted))

    if label is None and plot is True: 
        
        plt.scatter(x_sorted, ecdf, alpha = 0.7)

    
    elif label is not None and plot is True: 
        
        plt.scatter(x_sorted, ecdf, alpha = 0.7, label = label)

    else: 
    	pass
        
    return x_sorted, ecdf


def freedman_diaconis_rule(arr): 
	"""
	Calculates the number of bins for a histogram using the Freedman-Diaconis Rule. 

	Modified from https://github.com/justinbois/bebi103/blob/master/bebi103/viz.py
	
	"""
	h = 2* (np.percentile(arr, q=75) - np.percentile(arr, q = 25))/ np.cbrt(len(arr))

	if h == 0.0:
		n_bins = 3
	else:
		n_bins = int(np.ceil(arr.max() - arr.min()) / h)

	return n_bins


def l1_norm(arr1, arr2):
	'''
	Compute the L1-norm between two histograms.
	It uses the Freedman-Diaconis criterion to determine the number of bins. 

	It will be positive if the mean(arr2) > mean(arr1) following the convention
	from PopAlign.

	Modified from https://github.com/thomsonlab/popalign/blob/master/popalign/popalign.py
	
	Parameters
	----------
	arr1 (array-like)
		Distribution of gene for population 1. 
	arr2 (array-like)
		Distribution of gene for population 2. 

	Returns 
	-------
	l1_score(float)
		L1 norm between normalized histograms of gene distributions. 

	Example
	-------
	import numpy as np
	from sc_utils import sc

	x = np.random.normal(loc = 0, size = 100)
	y = np.random.normal(loc = 3, size = 100)

	sc.l1_norm(x, y)
	>>>1.46
	'''

	if len(arr1) == len(arr2): 
		nbins = freedman_diaconis_rule(arr1)

	else: 
		nbins_1 = freedman_diaconis_rule(arr1)
		nbins_2 = freedman_diaconis_rule(arr2)

		nbins = int((nbins_1 + nbins_2)/2)


	max1, max2 = np.max(arr1), np.max(arr2) # get max values from the two subpopulations
	max_ = max(max1,max2) # get max value to define histogram range
	if max_ == 0:
		return 0
	else:
		b1, be1 = np.histogram(arr1, bins=nbins, range=(0,max_)) # compute histogram bars
		b2, be2 = np.histogram(arr2, bins=nbins, range=(0,max_)) # compute histogram bars
		b1 = b1/len(arr1) # scale bin values
		b2 = b2/len(arr2) # scale bin values
		if arr1.mean()>=arr2.mean(): # sign l1-norm value based on mean difference
			l1_score = -np.linalg.norm(b1-b2, ord=1)
			return l1_score
		else:
			l1_score = np.linalg.norm(b1-b2, ord=1)
			return l1_score
