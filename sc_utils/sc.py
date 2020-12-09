# Import workhorses
import scipy.io as sio
import scipy.stats as st
import scanpy as sc
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

from sklearn.utils import sparsefuncs

def check_sparsity(adata): 
	"Returns the percentage of zeros in dataset."

	if not sparse.isspmatrix_csr(adata.X):
		adata.X = sparse.csr_matrix(adata.X)

	sparsity = (1 - adata.X.data.shape[0] / (adata.n_obs*adata.n_vars))*100

	return sparsity 


def get_count_stats(adata, mt_prefix = 'MT-', ribo_prefix = None):

	"""
	Returns an adata with extra columns in its `obs` object 
	for the number of counts per cell `n_counts` (and log10 (counts) ),
	and the number of expressed genes in each cell `n_genes`.
	Additionally it can get the fraction of mitochondrial and ribosomal
	genes if prefixes are provided.
	
	Notes: Expects adata in sparse.csr_matrix format. 

	mt_prefix (str, default = None): 
		Prefix to match mitochondrial genes. For human is `MT-`
		and for mouse is `mt`. 

	ribo_prefix()
		For human the prefixes are ('RPS', 'RPL'). 
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



def lognorm_cells(adata, scaling_factor = 1e4, log = True): 

	"""
	Cell count normalization as in sc.pp.normalize_total.
	Expects count matrix in sparse.csr_matrix format. 

	Each gene's expression value in a given cell is given by : 

	g_i = \mathrm{ln} ( \frac{g_i \times \beta }{\sum g_i} + 1 )
	
	where β is the scaling factor. 

	Note of caution: normalization is done inplace.

	Params
	------
	adata (ad.AnnData): 
		Gene count matrix with cell and gene annotations. 
	
	scaling_factor(float, default = 1e4)
		Factor to scale gene counts to represent the counts in 
		the cell. If scaling_factor =1e4, the values will 
		represent counts per million. 
	
	log (bool, default = True)
		Optional argument to allow for returning the scaled cells 
		without normalizing. 
	
	Returns 
	-------
	adata (ad.AnnData): 
		Anndata with normalized and log transformed count matrix. 
	"""
	

	if not sparse.isspmatrix_csr(adata.X):
		adata.X = sparse.csr_matrix(adata.X)

	# Get total counts per cell from `obs` df
	if 'n_counts' in adata.obs.columns:
		counts = adata.obs.n_counts.values

	else: 
		counts = adata.X.sum(axis = 1).flatten()

	# Convert to numpy matrix to array to be able to flatten
	scaled_counts = np.array(counts).flatten() / scaling_factor

	sparsefuncs.inplace_csr_row_scale(adata.X, 1/scaled_counts)

	# Call the log1p() method on the csr_matrix 
	if log:
		
		adata.X = adata.X.log1p()

	return adata

@tz.curry
def cv_filter(adata, min_mean = 0.025, min_cv= 1, transform = False):

	"""
	Performs the Coefficient of Variation filtering according 
	to the Poisson / Binomial counting statistics. The model assumes 
	the coefficient of variation per gene is given by : 

	\mathrm{log} (CV) \approx - \frac{1}{2}\mathrm{log} (\mu) + \epsilon
 

	The values will be computed assuming a normalized and 
	log-scaled count matrix. 
	
	Params 
	------
	min_mean (float, default = None). 
		Lower bound cutoff for the mean of the gene feature. 
		Recommended value = 0.01. 
	
	min_cv (float, default = None)
		Lower bound for the coefficient of variation of the 
		gene feature. Recommended value 1. 
	
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

	adata.var = pd.concat(
	    [adata.var, df_gene_stats], 
	    axis = 1
	)

	from scipy.stats import linregress

	#t0_lr = time.time()
	slope, intercept, r, pval, stderr = linregress(log_mean, log_cv)
	#t_end_lr = time.time()

	# Check that slope is approx -1/2
	print(f'The slope of the model is {np.round(slope,3)}.')
	#print(f'The linear regression took : {(t_end_lr - t0_lr)} seconds.\n')

	poisson_prediction_cv = slope*log_mean + intercept

	# Binary array of highly variable genes 
	gene_sel = log_cv > poisson_prediction_cv
	
	adata.var['highly_variable'] = gene_sel.astype(int)

	if min_mean and min_cv is not None: 
		adata_filt = adata[:,(adata.var.highly_variable == True)&\
							 (adata.var['mean'] > min_mean)&\
							 (adata.var['cv'] > min_cv)].copy()
	else: 
		adata_filt = adata[:, adata.var.highly_variable == True].copy()
	
	if transform: 
		return adata_filt

	else: 
		return adata



def score_cells_percentile(val, percs): 
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
        score = 1
    elif val < low_perc: 
        score = -1
    else: 
        score = 0
    return score

def score_cells_along_component(latent_dim, percs = (10, 90)): 
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
    low_perc, hi_perc = np.percentile(latent_dim, (10, 90))
    
    scores = [score_cells_percentile(p, (low_perc, hi_perc)) for p in df_proj['latent_3']]
    
    return scores



def get_scanpy_deg_report_df(adata, groups = ('-1','1')):
    """
    Returns a report dataframe of differentially expressed genes. 
    It expects an adata with a report dictionary from the output of
    scanpy.tl.rank_genes_groups(). 
    
    Params 
    ------
    adata (ad.AnnData)
        AnnData with rank_genes_groups dictionary in `.uns` object.
        Ideally, this adata would only contain "prototype" cells,
        that is, the cells on the extremes of a given component. 
        
    groups (tuple, default = (-1,1))
        Tuple of groups for which to extract the DEG results.
    
    """
    
    deg_result_dict = adata.uns['rank_genes_groups']
    
    # Make a report dataframe. 
    df_report = pd.DataFrame()

    for g in groups:
        df = pd.DataFrame(
            np.vstack(
                [deg_result_dict["names"][g],
                 deg_result_dict["logfoldchanges"][g],
                 deg_result_dict["pvals_adj"][g]]
            ).T,
            columns=["gene_name", "log_fc", "pval_adj"],
        )

        df["group"] = g

        df_report = pd.concat([df_report, df])
    
    return df_report


def annot_genes(gene, gene_sets_tuple):
	"""
	Helper function to annotate a given gene in prototype cells
	enriched genes.
	Returns -1 if gene is in bottom prototype, 1 if in top prototype. 
	"""

	gene_set_low, gene_set_hi = gene_sets_tuple
    
	if gene_name in gene_set_low:
	    y = -1
	elif gene_name in gene_set_hi: 
	    y = 1
	else: 
	    y = 0

	return y

def run_fisher_test_go(df_report_deg, path, n_top = 5, organism = 'human'):
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
    
    #Load Gene Ontology dataset (from QuickGO annotation)
    df_go =  pd.read_csv(path + 'go/go_lite_' + organism +'.csv')

    # Get gene names of DEG genes
    genes_disease = df_report_deg[df_report_deg['group']== '-1']['gene_name'].values
    genes_healthy = df_report_deg[df_report_deg['group'] == '1']['gene_name'].values
    
    gene_sets = (genes_disease, genes_healthy)
    
    # Annotate genes as prototypes:
    # These annotations will help the Fisher enrichment test function
    df_go['prototype'] = [annot_genes(g, gene_sets) for g in df_go.gene_name.values]

    # Compute FET for bottom and top prototypes

    df_enrichment_lo_prot = fisher_enrichment_test(
        df_annot = df_go, 
        clus_col_name = 'prototype',
        annotation = 'GO NAME', 
        cluster = -1,
        n_top = n_top
    )

    df_enrichment_top_prot = fisher_enrichment_test(
        df_annot = df_go, 
        clus_col_name = 'prototype',
        annotation = 'GO NAME', 
        cluster = 1, 
        n_top = 10
    )
    
    df_enrichment_report = pd.concat([df_enrichment_lo_prot,
                                     df_enrichment_top_prot])
    
    df_enrichment_report = df_enrichment_report.\
                        sort_values_by(['prototype', 'pval']).reset_index(drop = True)
    
    df_enrichment_report = df_enrichment_report.rename(columns = {'clusters':'prototype'})

    # Get GO categories for differentially expressed genes
    # i.e. in prototypes -1 or 1 (not in 0).
    try:
        df_go_red = df_go[df_go['prototype'].isin([-1,1])]
    except: 
        df_go_red = df_go[df_go['prototype'].isin(['-1','1'])]
        
    # Load Transcription Factor gene names
    path_to_tfs = path + '../trns/' + organism + '_trn/'+ organism + '_tfs.csv'
    
    tf = pd.read_csv(path_to_tfs)
    
    # Annotate if DEG genes are TFs
    df_go_red['is_tf'] = df_go_red.gene_name.apply(
        lambda x: 1 if x in tf['gene_name'].unique() else 0
    )
    
    return df_enrichment_report, df_go_red


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



def fisher_enrichment_test(df_annot, annotation,
	cluster, clus_col_name = 'cluster_labels', n_top = 5): 
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
    
    col_clus_name (str)
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
    df_clus = df_test[df_test[clus_col_name] == cluster]

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

    df_report['cluster'] = cluster
    df_report['annot'] = annotation

    return df_report 



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