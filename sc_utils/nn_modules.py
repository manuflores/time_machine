import gzip 
import numpy as np 
from scipy import sparse
import pandas as pd
import tqdm 
import re
import collections
import anndata as ad
import multiprocessing as mp

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.distributions as td

import torchvision
import torchvision.transforms as transforms 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, IterableDataset, DataLoader


def try_gpu(i=0):  
    """
    Return gpu(i) if exists, otherwise return cpu().

    Extracted from https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """
    Return all available GPUs, or [cpu(),] if no GPU exists.

    Extracted from https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py
    """
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]



def initialize_network_weights(net, method = 'kaiming', seed = 4): 
    """
    Initialize fully connected and convolutional layers' weights
    using the Kaiming (He) or Xavier method. 
    This method is recommended for ReLU / SELU based activations.
    """

    torch.manual_seed(seed)

    if method == 'kaiming': 
        for module in net.modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)): 
                nn.init.kaiming_uniform_(module.weight)
                nn.init.uniform_(module.bias)

            elif isinstance(module, (nn.GRU, nn.LSTM)): 
                for name, param in module.named_parameters():                    
                    if 'bias' in name : 
                        nn.init.uniform_(param)
                    elif  'weight' in name:
                        nn.init.kaiming_uniform_(param)
                    else:
                        pass

            else: 
                pass


    elif method == 'xavier':
        for module in net.modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)): 
                nn.init.xavier_uniform_(module.weight)
                nn.init.uniform_(module.bias)

            elif isinstance(module, (nn.GRU, nn.LSTM)): 
                for name, param in module.named_parameters():                    
                    if 'bias' in name : 
                        nn.init.uniform_(param)
                    elif  'weight' in name:
                        nn.init.xavier_uniform_(param)
                    else:
                        pass

            else: 
                pass

    elif method == 'xavier_normal':
        for module in net.modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)): 
                nn.init.xavier_normal_(module.weight)
                nn.init.uniform_(module.bias)

            elif isinstance(module, (nn.GRU, nn.LSTM)): 
                for name, param in module.named_parameters():                    
                    if 'bias' in name : 
                        nn.init.uniform_(param)
                    elif  'weight' in name:
                        nn.init.xavier_normal_(param)
                    else:
                        pass

            else: 
                pass


    else: 
        print('Method not found. Only valid for kaiming or xavier initialization.')

    return net



class StreamingDataset(IterableDataset): 
    """
    Allows to stream a dataset in tsv or csv format. 
    It inherits the torch.utils.data.IterableDataset for handling
    large streams of data. 

    This object is designed to work on text files containing numerical 
    data only. In this sense, a separate file for annotation would be used. 
    This is also designed to work with unsupervised learning problems, e.g.
    autoencoders, as there is no label or other dataset needed for the direct computation,
    though this could be easily extended. 
    
    Params 
    -------
    
    file_path (str)
        Path to file containing the dataset. 

   transform (torch.transform)
        Torch transform. For example see the ToTensor() transform. 
    
    delim (str, default = '\t')
        File delimiter. It defaults to a tab. 

    unzip (bool, default = False)
        If set to True, it uses the gzip library to unzip the dataset. 
    
    
    Returns 
    -------
    line_generator (generator)
        A generator object that contains a numpy array of each line
        in the input dataset.

    """

    def __init__(self, file_path, transform = None, delim = '\t', unzip = False): 
        self.file_path = file_path
        self.transform = transform
        self.delim = delim
        self.unzip = unzip
    
    def line_to_arr(self, line):
        "Converts a line of a text file into a numpy array."
        lst = [np.float(elem) for elem in line.rstrip().split(self.delim)]        
        arr = np.array(lst)
        return arr 
    
    
    def __iter__(self):
        
        # Unzip as part of the streaming process
        if self.unzip: 
            file_itr = gzip.open(self.file_path, mode = 'rt')
        
        # Read plain text file. 
        else: 
            file_itr = open(self.file_path)
        
        # Map the preprocessing function to each line 
        line_generator = map(self.line_to_arr, file_itr)
        
        return line_generator



class StreamingDataset_mp(IterableDataset): 
    """
    Allows to stream a dataset in tsv or csv format using multiprocessing. 
    This is still a work in process given that we have found empirically 
    that it doesn't actually speed up the training. 

    We recommend using the StreamingDataset function. 

    It inherits the torch.utils.data.IterableDataset
    for handling large datasets in a single machine. 

    This object is designed to work on text files containing numerical 
    data only. In this sense, a separate file for annotation would be used. 
    This is also designed to work with autoencoders as there is no label
    or other dataset needed for the direct computation, though this could be
     easily extended. 
    
    Params 
    -------
    
    file_path (str)
    	Path to file containing the dataset. 

    file_size (int)
    	Number of lines in txt file. 

    transform (torch.transform)
    	Torch transform. Examples are a ToTensor transform.  
    
    delim (str, default = '\t')
        File delimiter. It defaults to a tab. 
	
	unzip (bool, default = False)
		If set to True, it uses the gzip library to unzip the dataset. 
    
    
    Returns 
    -------
	file_itr or file_gen (iterator or generator)
		A generator object that contains a numpy array of each line
		in the input dataset.

    """

    def __init__(self, file_path, file_size, transform = None, delim = '\t', unzip = False): 
        self.file_path = file_path
        self.file_size = file_size
        self.transform = transform
        self.delim = delim
        self.unzip = unzip
    
    def line_to_arr(self, line):
        "Converts a line of a text file into a numpy array."
        lst = [np.float(elem) for elem in line.rstrip().split(self.delim)]        
        arr = np.array(lst)
        return arr 
    
    def yield_lines(self, file, indices): 
    	"Returns a generator expression that returns data over specified indices"
    	return (line for ix, line in enumerate(file) if ix in indices)

    
    def __iter__(self):
        
    	worker_info = torch.utils.data.get_worker_info()


    	if worker_info is None: # single-process data loading 

    		# Check if unzipping is part of the streaming process
    		if self.unzip: 
    			file_itr = gzip.open(self.file_path, mode = 'rt')
    			line_itr = map(self.line_to_array, file_itr)
    			
    		# Read plain txt file. 
    		else: 
    			file_itr = open(self.file_path)
    			line_itr = map(self.line_to_array, file_itr)

    		return line_itr

    	else: # in a worker process
    		n_workers = worker_info.num_workers
    		worker_id = worker_info.id
    		# print(f'Worker id {worker_id}')

    		# Lines to ingest per worker 
    		per_worker = int(math.ceil(self.file_size / n_workers))

    		#print(f' lines per worker {per_worker}')

    		# index starts at 0 if worker_id ==0
    		iter_start = worker_id*per_worker 

    		iter_end = min(iter_start + per_worker, self.file_size)

    		indices = np.arange(iter_start, iter_end)


    		if self.unzip:
    			file_itr = gzip.open(self.file_path, mode = 'rt')
    			line_itr = map(self.line_to_arr, file_itr)
    			file_gen = self.yield_lines(line_itr, indices)

    		else: 
    			file_itr = open(self.file_path)
    			line_itr = map(self.line_to_arr, file_itr)
    			file_gen = self.yield_lines(line_itr, indices)

    		return file_gen 



class protein_dataset(Dataset): 

	"PFAM protein dataset."

	def __init__(self, fname, transform = None): 

		#self.sequence_df = pd.read_csv(fname)

		self.files = glob.glob(filename)

		self.transform = transform 

		# Initialize aminoacid vocabulary
		codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         		 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        # Initialize aminoacid to idx dictionary
		aa_dict = dict(zip(codes, np.arange(1, 21)))

		self.aa_dict = aa_dict
	


	def seq2hot_matrix(self, line): 

		"""
		Preprocessing function to generate a one-hot-encoded matrix 
		of size (100, 21) from a sequence of aminoacids. 

		"""

		encoded_seq = [self.aa_dict.get(aa, 0) for aa in line]

		padded_seq = pad_sequences(
    		encoded_seq,
    		maxlen = 100,
    		padding = 'post'
		)

		one_hot_matrix = to_categorical(padded_seq)

		return one_hot_matrix 


	def __getitem__(self, ix): 

		if torch.is_tensor(ix):
			ix = ix.to_list()


		seq = sequence_df.iloc[ix, :].sequence.values




class StreamingProteinDataset(IterableDataset): 

	def __init__(self,  file_path, annot_fname, batch_size, transform = None):

		self.file_path = file_path
		self.files = glob.glob(file_path)
		self.transform = transform
		self.unzip = unzip
		self.batch_size = batch_size

		# Initialize aminoacid vocabulary
		codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         		 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        # Initialize aminoacid to idx dictionary
		aa_dict = dict(zip(codes, np.arange(1, 21)))

		self.aa_dict = aa_dict 


		self.annotation_df = pd.read_csv(file_path, annot_fname)
		self.label_list = self.annotation_df['labels'].to_numpy()
		self.list_IDs = self.annotation_df.index.to_numpy()


	def __len__(self): 

		return len(self.list_IDs)

	def seq2hot_matrix(self, line): 

		"""
		Preprocessing function to generate a one-hot-encoded matrix 
		of size (100, 21) from a sequence of aminoacids. 

		"""

		encoded_seq = [self.aa_dict.get(aa, 0) for aa in line]

		padded_seq = pad_sequences(
    		encoded_seq,
    		maxlen = 100,
    		padding = 'post'
		)

		one_hot_matrix = to_categorical(padded_seq)

		return one_hot_matrix 


	def __iter__(self): 


		# Unzip as part of the streaming process
		if self.unzip: 
			file_iter = gzip.open(self.file_name, mode = 'rt')

		# Read plain txt file 
		else: 
			file_iter = open(self.file_name)

		# Map the preprocessing function to each line 
		seq_generator = map(self.seq2hot_matrix, file_iter)

		return seq_generator



class VariationalAutoencoder(nn.Module): 

    def __init__(self, dims, beta = 1, recon_loss = 'BCE'):
        """
        Variational autoencoder (VAE) module with single hidden layer. 
        Contains functions to reconstruct a dataset and 
        project it into a latent space.

        Params 
        ------
        dims (array-like):
            Dimensions of the networks given by the number of neurons
            of the form [input_dim, hidden_dim_1, ..., hidden_dim_n, latent_dim], 
            where `input_dim` is the number of features in the dataset. 

            Note: The encoder and decoder will have a symmetric architecture.

        recon_loss (str, defult = 'BCE')
            Reconstruction loss. Avaiable loss functions are 
            binary cross entropy ('BCE') and mean squared error ('MSE'). 

            We have empirically found that using BCE on minmax normalized
            data gives good results. 

        dims (array-like):
                Dimensions of the networks given by the number of neurons
                of the form [input_dim, hidden_dim_1, ..., hidden_dim_n, latent_dim].
                The encoder and decoder will have a symmetric architecture.
            
        """
        
        super(VariationalAutoencoder, self).__init__()

        
        self.input_dim = dims[0]
        #self.output_dim = dims[0]
        self.embedding_dim = dims[-1]

        # ENCODER

        # Start range from 1 so that dims[i-1] = dims[0]
        hidden_layers_encoder = [
            BnLinear(dims[i-1], dims[i]) for i in range(1, len(dims[:-1]))
        ] 

        self.encoder_hidden = nn.ModuleList(hidden_layers_encoder)

        # Stochastic layers 
        self.mu = BnLinear(dims[-2], self.embedding_dim)
        self.log_var = BnLinear(dims[-2], self.embedding_dim)


        # DECODER
        dims_dec = dims[::-1] # []

        hidden_layers_decoder = [
            BnLinear(dims_dec[i-1], dims_dec[i]) for i in range(1, len(dims_dec[:-1]))
        ]

        self.decoder_hidden = nn.ModuleList(hidden_layers_decoder)

        self.reconstruction_layer = BnLinear(dec_dims[-2], self.input_dim)

        # Activation functions 
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        #self.tanh= nn.Tanh()

    def encoder(self, x): 
        """
        Encode a batch of samples and return posterior parameters 
        mu and logvar for each point. 

        Attempts to generate probability distribution P(z|x)
        from the data by fitting a variation distribution Q_φ(z|x).
        Returns the two parameters of the distributon (µ, log σ²).

        """
        for fc_layer in self.encoder_hidden: 
            x = fc_layer(x)
            x = F.tanh(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar 

    def decoder(self, z): 
        """
        Decodes a batch of latent variables.

        Generative network. Generates samples from the original data 
        distribution P(x) by approximating p_θ(x|z). It uses a Sigmoid activation 
        for the output layer, so input data must be normalized between 0 and 1
        (e.g. min-max normalized).
        """
        for fc_layer in self.decoder_hidden: 
            x = fc_layer(x)
            x = F.tanh(x)

        x = self.reconstruction_layer(x)
        x = self.sigmoid(x)

        return x

    def reparam(self, mu, logvar):
        """
        Reparametrization trick to sample z values. 
        This is a stochastic procedure, and returns 
        the mode during evaluation.
        """

        if self.training:

            std = logvar.mul(0.5).exp_()

            epsilon = Variable(torch.randn(mu.size()), requires_grad = False)

            if mu.is_cuda: 
                epsilon = epsilon.cuda()

            #z = ϵ * σ + µ 

            z = eps.mul(std).add_(mu)

            return z

        else:
            return mu

    def forward(self, x):
        """
        Forward pass through Encoder-Decoder.
        """
        
        mu, logvar = self.encoder(x.view(-1, self.input_size))
        z = self.reparam(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def loss(self, reconstruction, x, mu, logvar):

        """
        Variationa loss, i.e. evidence lower bound (ELBO)
        It uses closed form of KL divergence between two Gaussians. 

        Params 
        ------
        x (torch.tensor)
            Minibatch of input data. 

        mu (torch.tensor)
            Output of mean of a stochastic gaussian layer.
            Used to compute KL-divergence. 

        logvar(torch.tensor)
            Output of log(σ^2) of a stochastic gaussian layer.
            Used to compute KL-divergence. 


        Returns
        -------
        variational_loss(torch.tensor)
            Sum of KL divergence plus reconstruction loss. 

        """
        
        if self.recon_loss == 'BCE': 

            reconstruction_loss = torch.nn.functional.binary_cross_entropy(
                reconstruction, x.view(-1, self.input_size)
            )

        elif self.recon_loss == 'MSE': 


            reconstruction_loss = torch.nn.functional.mse_loss(
                reconstruction, x.view(-1, self.input_size)
            )

        else : 
            print('Loss function provided is not available.')


        # Gaussian - Gaussian KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #  Normalize by same number of elems as in reconstruction
        KLD /= x.view(-1, self.input_size).data.shape[0] * self.input_size

        variational_loss = reconstruction_loss + self.beta*KLD

        return variational_loss

    def get_z(self, x):
        
        """
        Encode a batch of data points into their latent representation z. 
        """

        mu, logvar = self.encoder(x.view(-1, self.input_size))

        z = self.reparam(mu, logvar)

        return z

    def project_data_into_latent_cell(self, data_loader): 
        """
        Generator function to project dataset into latent space. 
        
        Params 
        ------
        data_loader (torch.DataLoader)
            DataLoader which handles the batches and parallelization. 
            
        n_feats (int)
            Number of dimensions of original dataset.
        
        model (nn.Module)
            Neural network model to be used for inference. 
            
        Returns (yields)
        -------
        encoded_sample (array-like generator)
            Generator single encoded sample in a numpy array format. 
        """
        
        cuda = torch.cuda.is_available()

        # Iterate through all of the batches in the DataLoader
        for batch_x in tqdm.tqdm(data_loader): 

            if cuda: 
                batch_x = batch_x.cuda()
            
            # Reshape to eliminate batch dimension 
            batch_x = batch_x.view(-1, self.input_dim)
            
            # Project into latent space and convert tensor to numpy array
            if cuda: 
                batch_x_preds = model.get_z(batch_x.float()).cpu().detach().numpy()
            else:                 
                batch_x_preds = model.get_z(batch_x.float()).detach().numpy()
            
            # For each sample decoded yield the line reshaped
            # to only a single array of size (latent_dim)
            for x in batch_x_preds:
                encoded_sample = x.reshape(self.embedding_dim)
                
                yield encoded_sample


    def get_gaussian_samples(self, x, n_samples = 10):
        """
        Generates n_samples from the posterior distribution 
        p(z|x) ~ Normal (µ (x), diag(σ(x))) from a single data point x.
        This function is designed to be used after training. 
        """
        x = x.view(-1, self.input_size)

        mu, log_var = self.encoder(x)

        var = log_var.exp_()

        cov_mat = torch.diag(var.flatten())

        gaussian = td.MultivariateNormal(mu, cov_mat)

        gaussian_samples = gaussian.sample((n_samples,))

        return gaussian_samples



class VAE(nn.Module):

    """
    Variational autoencoder (VAE) module with single hidden layer. 
    Contains functions to reconstruct a dataset and 
    project it into a latent space.

    Note: Old VAE function, use VariationalAutoencoder now. 

    Params 
    ------
    input_size (int)
        Number of features of input dataset. 

    latent_dim (int)
        Number of features in the embedding space of the AE. 

    recon_loss (str, defult = 'BCE')
        Reconstruction loss. Avaiable loss functions are 
        binary cross entropy and mean squared error. 

        We have empirically found that using BCE on minmax normalized
        data gives good results. 
        
    """

    def __init__(self, input_size, hidden_size, latent_dim, beta = 1, recon_loss = 'BCE'):
        
        super(VAE, self).__init__()

        # Extra layers regularization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p = 0.1)


        # Encoder layers
        self.en1 = nn.Linear(input_size, hidden_size)
        self.mu_ = nn.Linear(hidden_size, latent_dim)
        self.std_ = nn.Linear(hidden_size, latent_dim)

        # Decoder layers
        self.de1 = nn.Linear(latent_dim, hidden_size)
        self.de2 = nn.Linear(hidden_size, input_size)
        
        self.beta = beta
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.input_size = input_size
        self.recon_loss = recon_loss

    def encoder(self, x):
        """
        Encode a batch of samples and return posterior parameters 
        mu_ and std_ for each point. 
        """
        x = self.en1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        mu = self.mu_(x)
        logvar = self.std_(x)

        return mu, logvar

    def decoder(self, z):
        """
        Decode a batch of latent variables.
        """
        x = self.de1(z)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.de2(x)
        x = self.sigmoid(x)

        return x

    def reparam(self, mu, logvar):
        """
        Reparametrization trick to sample z values. 
        This is a stochastic procedure, and returns 
        the mode during evaluation.
        
        """

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())

            return eps.mul(std).add_(mu)

        else:
            return mu


    def forward(self, x):
        """
        Takes a batch of samples, encodes them, and then decodes 
        them again for comparison. 
        """
        
        mu, logvar = self.encoder(x.view(-1, self.input_size))
        z = self.reparam(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def loss(self, reconstruction, x, mu, logvar):

        """
        ELBO assuming entries of x are binary variables, or can be approximated by it.
        It uses closed form of KL divergence between two Gaussians. 

        Params 
        ------

        Returns
        -------

        """
        
        if self.recon_loss == 'BCE': 

            reconstruction_loss = torch.nn.functional.binary_cross_entropy(
                reconstruction, x.view(-1, self.input_size)
            )

        elif self.recon_loss == 'MSE': 


            reconstruction_loss = torch.nn.functional.mse_loss(
                reconstruction, x.view(-1, self.input_size)
            )

        else : 
            print('Loss function provided is not available.')


        # Gaussian - Gaussian KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #  Normalize by same number of elems as in reconstruction
        KLD /= x.view(-1, self.input_size).data.shape[0] * self.input_size

        return reconstruction_loss + self.beta*KLD
        

    def get_z(self, x):
        
        """
        Encode a batch of data points into their latent representation z. 
        """

        mu, logvar = self.encoder(x.view(-1, self.input_size))

        z = self.reparam(mu, logvar)

        return z

    def get_gaussian_samples(self, x, n_samples = 10):
        """
        Generates n_samples from the posterior distribution 
        p(z|x) ~ Normal (µ (x), diag(σ(x))) from a single data point x.
        This function is designed to be used after training. 
        """
        x = x.view(-1, self.input_size)

        mu, log_var = self.encoder(x)

        var = log_var.exp_()

        cov_mat = torch.diag(var.flatten())

        gaussian = td.MultivariateNormal(mu, cov_mat)

        gaussian_samples = gaussian.sample((n_samples,))

        return gaussian_samples





class VAE_(nn.Module):

    """
    Variational autoencoder (VAE) module with single hidden layer. 
    Contains functions to reconstruct a dataset and 
    project it into a latent space. 

    Old VAE module, use the VAE() module instead.

    Params 
    ------
    input_size (int)
        Number of features of input dataset. 

    latent_dim (int)
        Number of features in the embedding space of the AE. 

    recon_loss (str, defult = 'BCE')
        Reconstruction loss. Avaiable loss functions are 
        binary cross entropy and mean squared error. 

        We have empirically found that using BCE on minmax normalized
        data gives good results. 
        
    """

    def __init__(self, input_size, hidden_size, latent_dim, recon_loss = 'BCE'):
        
        super(VAE_, self).__init__()

        # Encoder layers
        self.en1 = nn.Linear(input_size, hidden_size)

        self.mu_ = nn.Linear(hidden_size, latent_dim)
        self.std_ = nn.Linear(hidden_size, latent_dim)

        # Decoder layers
        self.de1 = nn.Linear(latent_dim, hidden_size)
        self.de2 = nn.Linear(hidden_size, input_size)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.input_size = input_size
        self.recon_loss = recon_loss

    def encode(self, x):
        """
        Encode a batch of samples and return posterior parameters 
        mu_ and std_ for each point. 
        """

        h1 = self.relu(self.en1(x))

        return self.mu_(h1), self.std_(h1)

    def decode(self, z):
        """
        Decode a batch of latent variables. 
        """

        h2 = self.relu(self.de1(z))
        return self.sigmoid(self.de2(h2))

    def reparam(self, mu, logvar):
        """
        Reparametrization trick to sample z values. 
        This is a stochastic procedure, and returns the mode
        during evaluation.
        
        """

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())

            return eps.mul(std).add_(mu)

        else:
            return mu

    def forward(self, x):
        """
        Takes a batch of samples, encodes them, and then decodes 
        them again for comparison. 
        """
        
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparam(mu, logvar)

        return self.decode(z), mu, logvar

    def loss(self, reconstruction, x, mu, logvar, beta= 1):

        """
        ELBO assuming entries of x are binary variables, or can be approximated by it.
        It uses closed form of KL divergence between two Gaussians. 

        Params 
        ------

        Returns
        -------

        """
        
        if self.recon_loss == 'BCE': 

            reconstruction_loss = torch.nn.functional.binary_cross_entropy(
                reconstruction, x.view(-1, self.input_size)
            )

        elif self.recon_loss == 'MSE': 


            reconstruction_loss = torch.nn.functional.mse_loss(
                reconstruction, x.view(-1, self.input_size)
            )

        else : 
            print('Loss function provided is not available.')


        # Gaussian - Gaussian KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #  Normalize by same number of elems as in reconstruction
        KLD /= x.view(-1, self.input_size).data.shape[0] * self.input_size

        return reconstruction_loss + beta*KLD
        

    def get_z(self, x):
        
        """
        Encode a batch of data points into their latent representation z. 
        """

        mu, logvar = self.encode(x.view(-1, self.input_size))

        return self.reparam(mu, logvar)



class ConvAE(nn.Module): 
    
    def __init__(self, latent_dim): 
        super(ConvAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Input size : (batch_size, 4, 60, 60)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5,stride = 2, padding = 1), # Output size of (30 x 30) 
            nn.SELU(), 
            nn.BatchNorm2d(16), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 4, 4, stride = 1,padding = 1), # Output size of (28 x 28)
            nn.SELU(), 
            nn.BatchNorm2d(4), 
            nn.MaxPool2d(2,2, return_indices = True)
        )
        
        self.unpool = nn.MaxUnpool2d(2, stride = 2, padding = 0)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride = 2), 
            nn.SELU(), 
            nn.BatchNorm2d(16), 
            nn.ConvTranspose2d(16, 4, 2, stride = 2),
            nn.SELU())
     
    def forward(self, x): 
        print(x.size())
        
        out, indices = self.encoder(x)
        
        out = self.unpool(self, indices)
        out = self.decoder(out)
        
        print(out.size())
        
        return out 
    

class ConvAutoencoder(nn.Module): 

    """
    Convolutional autoencoder. 
    
    Params
    ------
    
    
    Returns 
    -------
    
    """

    def __init__(self, latent_dim): 
        super(ConvAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim

        # ---- ENCODER LAYERS -------

        # Conv layer (depth from 4 -> 8)
        # Output size of (30 x 30) assuming n_h = n_w = 60
        self.conv1 = nn.Conv2d(
            in_channels = 4,
            out_channels = 16, 
            kernel_size = 5, 
            stride = 2, 
            padding = 2
        )

        # Output size of (28 x 28)
        self.conv2 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 4, 
            kernel_size = 4, 
            stride = 1,
            padding = 1
        )

        # Reduce resolution by half 
        # (n_h - k_h + s_h) / s_h
        self.pool = nn.MaxPool2d(
            kernel_size = 2, 
            stride = 2, 
            return_indices = True
        )
        
        self.enc_fc_1 = nn.Linear(14*14*4, 64)
        self.enc_fc_2 = nn.Linear(64, self.latent_dim)
        
        # ----- DECODER LAYERS -------
        
        self.dec_fc_1 = nn.Linear(self.latent_dim, 64)
        self.dec_fc_2 = nn.Linear(64, 14*14*4)
        
        self.t_conv1 = nn.ConvTranspose2d(
            in_channels = 4, 
            out_channels = 16, 
            kernel_size = 2, 
            stride = 2
        )

        self.t_conv2 = nn.ConvTranspose2d(
            in_channels = 16, 
            out_channels = 4, 
            kernel_size = 2,
            stride = 2
        )


    def encode(self, x): 

        "Encode a batch of samples into the latent space."

        # Conv layers 
        x = self.pool(F.selu(self.conv1(x)))
        x = self.pool(F.selu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, self.num_flat_features(x))
        
        # Final fully connected layers
        x = F.selu(self.enc_fc_1(x))
        x = F.selu(self.enc_fc_2(x))

        return x

    def decode(self, z): 
        "Decode a batch of encoded samples to the original dimension."
        
        # Fully connected layers 
        x = F.selu(self.dec_fc_1(z))
        x = F.selu(self.dec_fc_2(x))
        
        # Convolutional layers
        x = self.pool(F.selu(self.t_conv1(x)))
        x = self.pool(F.selu(self.t_conv2(x)))

        return x


    def forward(self, x): 

        code = self.encode(x)
        out = self.decode(code)

        return out #, code 
    
    
    def num_flat_features(self, x): 
        
        size = x.size()[1:] # Get all dims except batch
        num_features = 1
        
        for s in size: 
            num_features *=s
        
        return num_features


class convAutoencoder(nn.Module): 
    
    def __init__(self, latent_dim, image_size ): 
        super(Autoencoder, self).__init__()
        
        self.image_size = image_size 
        
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride = 2, padding = 1), # Output size of (30 x 30) 
            nn.SELU(), 
            #nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 4, 4, stride = 1,padding = 1), # Output size of (28 x 28)
            #nn.SELU(), 
            #nn.MaxPool2d(2,2, return_indices = True)
            #nn.Embedding(28*28*4, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            #nn.Linear(latent_dim, 28*28*4),
            nn.ConvTranspose2d(4, 16, 4, stride = 1, padding = 1, output_padding = 0), 
            nn.SELU(), 
            #nn.BatchNorm2d(16), 
            nn.ConvTranspose2d(16, 4, 5, stride = 2, padding =1,  output_padding = 1),
            nn.SELU()
        )
    
    def reshape(self, x):
        batch_size = x.shape[0]
        
        return x.view(batch_size, -1, image_size, image_size)
    
    def encode(self, x): 
        x = self.encoder(x)
        return x
    
    def decode(self, x): 
        x = self.decoder(x)
        return x
        
    def forward(self, x): 
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x 


class Autoencoder_2(nn.Module): 
    
    def __init__(self, latent_dim, image_size, batch_size): 
        super(Autoencoder, self).__init__()
        
        self.image_size = image_size 
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        
        self.conv1 = nn.Conv2d(4, 16, 5, stride = 2, padding = 1) # Output size of (30 x 30) 
        self.selu = nn.SELU()

        #nn.MaxPool2d(2, 2), 
        self.conv2 = nn.Conv2d(16, 4, 4, stride = 1,padding = 1) # Output size of (28 x 28)
        #nn.SELU(), 
        #nn.MaxPool2d(2,2, return_indices = True)
        #self.flatten()
        self.en_fc = nn.Linear(28*28*4, latent_dim)
        
        
        #self.decoder = nn.Sequential(
        self.dec_fc = nn.Linear(latent_dim, 28*28*4)
        #nn.SELU(),
        #self.reshape(),
        self.conv_t1 = nn.ConvTranspose2d(4, 16, 4, stride = 1, padding = 1, output_padding = 0)
        #nn.SELU(), 
        #nn.BatchNorm2d(16), 
        self.conv_t2 = nn.ConvTranspose2d(16, 4, 5, stride = 2, padding =1,  output_padding = 1)
        
    
    def flatten(self, x): 
        
        return x.view(self.batch_size, -1)
        
    def reshape(self, x):
        batch_size = x.shape[0]
        
        return x.view(batch_size, -1, self.image_size, self.image_size)
    
    def encode(self, x): 

        x = self.selu(self.conv1(x))
        x = self.selu(self.conv2(x))
        x = self.selu(self.en_fc(x.view(-1, 28*28*4)))
        return x
    
    def decode(self, x): 
        x = self.selu(self.dec_fc(x.view(-1, self.latent_dim)))
        x = self.selu(self.conv_t1(x.view(self.batch_size, 4, self.image_size, self.image_size)))

        x = self.selu(self.conv_t2(x))

        return x
        
    def forward(self, x): 
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x 


class ConvolutionalAutoencoder(nn.Module): 
    
    def __init__(self, latent_dim, image_size, batch_size): 
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.image_size = image_size 
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        
        self.conv1 = nn.Conv2d(4, 16, 5, stride = 2, padding = 1) # Output size of (30 x 30) 
        self.selu = nn.SELU()

        #nn.MaxPool2d(2, 2), 
        self.conv2 = nn.Conv2d(16, 4, 4, stride = 1,padding = 1) # Output size of (28 x 28)
        #nn.SELU(), 
        #nn.MaxPool2d(2,2, return_indices = True)
        #self.flatten()
        self.en_fc = nn.Linear(28*28*4, latent_dim)
        
        
        #self.decoder = nn.Sequential(
        self.dec_fc = nn.Linear(latent_dim, 28*28*4)
        #nn.SELU(),
        #self.reshape(), 
        self.conv_t1 = nn.ConvTranspose2d(4, 16, 4, stride = 1, padding = 1, output_padding = 0) 
        #nn.SELU(), 
        #nn.BatchNorm2d(16), 
        self.conv_t2 = nn.ConvTranspose2d(16, 4, 5, stride = 2, padding =1,  output_padding = 1)


    def encode(self, x): 

        x = self.selu(self.conv1(x))
        x = self.selu(self.conv2(x))
        x = self.en_fc(x.view(-1, 28*28*4))
        return x
    
    def decode(self, x): 
        x = self.selu(self.dec_fc(x.view(-1, self.latent_dim)))
        x = self.selu(self.conv_t1(x.view(self.batch_size, 4, 28, 28)))

        x = self.selu(self.conv_t2(x))
        
        return x
        
    def forward(self, x): 
        x = self.encode(x)
        x = self.decode(x)
        
        return x 


class ConvolutionalAutoencoder_(nn.Module): 
    
    def __init__(self, latent_dim, image_size, batch_size, in_channels): 
        super(ConvolutionalAutoencoder_, self).__init__()

        self.image_size = image_size 
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        # Conv layer attributes 
        self.kernel_size_1 = int(np.sqrt(image_size) * 1.5)
        self.kernel_size_2 = int(np.sqrt(image_size))

        self.out_channels_1 = int(in_channels * 4)
        self.out_channels_2 = int(in_channels * 8)
        
#         print('out channels 1', self.out_channels_1)
#         print('out channels 2', self.out_channels_2)
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()

        # This pooling layer reduces spatial resolution by a factor of 2
        self.pool = nn.MaxPool2d(2, 2, return_indices = True)

        # Conversely this pooling layer increases (resets) spatial resolution by factor of 2. 
        self.unpool = nn.MaxUnpool2d(2, 2)

        self.conv1 = nn.Conv2d(
            in_channels,
            self.out_channels_1,
            self.kernel_size_1,
            stride = 2,
            padding = 1
        ) 

        self.conv2 = nn.Conv2d(
            self.out_channels_1,
            self.out_channels_2, 
            self.kernel_size_2, 
            stride = 1,
            padding = 1
        )

        self.output_feat_1 = self.calculate_output_featmap_size(
            image_size, self.kernel_size_1, padding = 1, stride = 2
        )

        self.output_feat_2 = self.calculate_output_featmap_size(
            self.output_feat_1, self.kernel_size_2, padding = 1, stride = 1
        )

        # Divide by 2 because of max pooling layer 
        self.input_size_fc = int(np.ceil(self.output_feat_2 / 2))
        
        self.en_fc = nn.Linear(
            int((self.input_size_fc**2) * self.out_channels_2), latent_dim
        )

        self.dec_fc = nn.Linear(
            latent_dim,
            int((self.input_size_fc**2) * self.out_channels_2)
        )


        self.conv_t1 = nn.ConvTranspose2d(
            self.out_channels_2,
            self.out_channels_1,
            self.kernel_size_2, 
            stride = 1, 
            padding = 1, 
            output_padding = 0
        ) 


        self.conv_t2 = nn.ConvTranspose2d(
            self.out_channels_1,
            in_channels, 
            self.kernel_size_1, 
            stride = 2, 
            padding =1,
            output_padding = 0
        )


    def encode(self, x): 

        x = self.selu(self.conv1(x))
        
        #x, self.indices_1 = self.pool(x)
        
        x = self.selu(self.conv2(x))
        
        x, self.indices_2 = self.pool(x)
        
        
        x = self.en_fc(x.view(-1, int(self.input_size_fc**2 * self.out_channels_2)))

        return x

    def decode(self, x): 

        x = self.selu(self.dec_fc(x.view(-1, self.latent_dim)))

        
        x = self.unpool(
            x.view(
                self.batch_size,
                self.out_channels_2,
                self.input_size_fc,
                self.input_size_fc
                ),
            self.indices_2
        )

        x = self.selu(self.conv_t1(x))

#         x = self.unpool(
#             x,
#             self.indices_1
#         )

        
        try: 
            x = self.conv_t2(x)

        except: 
            conv_t2.output_padding = (1,1)
            x = conv_t2(x)

        x = self.sigmoid(x)
        return x

    def forward(self, input): 
        latent = self.encode(input)
        reconstructed = self.decode(latent)

        # Add extra padding in the last conv layer 
        # if input dimensions do not match output dimensions 
        if input.size != reconstructed.size: 
            self.conv_t2.output_padding = (1,1)

            latent = self.encode(input)
            reconstructed = self.decode(latent)

        return reconstructed

    def calculate_output_featmap_size(self, in_size, kernel_size, padding, stride): 
        """
        Helper function to calculate size of the output 
        of a conv layer for square images. 
        """
        return int(np.ceil(in_size - kernel_size + 2*padding + stride) / stride) 




class ConvolutionalAutoencoder_2(nn.Module): 

    """
    ConvAE using two pooling layers. WIP....
    """
    
    def __init__(self, latent_dim, image_size, batch_size, in_channels): 
        super(ConvolutionalAutoencoder_2, self).__init__()

        self.image_size = image_size 
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        # Conv layer attributes 
        self.kernel_size_1 = int(np.sqrt(image_size) * 1.5)
        self.kernel_size_2 = int(np.sqrt(image_size))

        self.out_channels_1 = int(in_channels * 4)
        self.out_channels_2 = int(in_channels * 8)
        
#         print('out channels 1', self.out_channels_1)
#         print('out channels 2', self.out_channels_2)
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()

        # This pooling layer reduces spatial resolution by a factor of 2
        self.pool = nn.MaxPool2d(2,2, return_indices = True)

        # Conversely this pooling layer increases (resets) spatial resolution by factor of 2. 
        self.unpool = nn.MaxUnpool2d(2, 2)

        self.conv1 = nn.Conv2d(
            in_channels,
            self.out_channels_1,
            self.kernel_size_1,
            stride = 2,
            padding = 1
        ) 

        self.conv2 = nn.Conv2d(
            self.out_channels_1,
            self.out_channels_2, 
            self.kernel_size_2, 
            stride = 1,
            padding = 1
        )

        self.output_feat_1 = self.calculate_output_featmap_size(
            image_size, self.kernel_size_1, padding = 1, stride = 2
        )

        self.output_pool_1 = self.output_feat_1 / 2


        self.output_feat_2 = self.calculate_output_featmap_size(
            self.output_pool_1, self.kernel_size_2, padding = 1, stride = 1
        )

        # Divide by 2 because of max pooling layer 
        self.input_size_fc = int(np.ceil(self.output_feat_2 / 2))
        
        print(self.input_size_fc)

        self.en_fc = nn.Linear(
            int((self.input_size_fc**2) * self.out_channels_2), latent_dim
        )

        self.dec_fc = nn.Linear(
            latent_dim,
            int((self.input_size_fc**2) * self.out_channels_2)
        )


        self.conv_t1 = nn.ConvTranspose2d(
            self.out_channels_2,
            self.out_channels_1,
            self.kernel_size_2, 
            stride = 1, 
            padding = 1, 
            output_padding = 0
        ) 


        self.conv_t2 = nn.ConvTranspose2d(
            self.out_channels_1,
            in_channels, 
            self.kernel_size_1, 
            stride = 2, 
            padding =1,
            output_padding = 1
        )


    def encode(self, x): 

        x = self.selu(self.conv1(x))
        
        x, self.indices_1 = self.pool(x)
        
        x = self.selu(self.conv2(x))
        
        x, self.indices_2 = self.pool(x)
        
        #print(x.size())
        
        x = self.en_fc(x.view(-1, int(self.input_size_fc**2 * self.out_channels_2)))

        return x

    def decode(self, x): 

        x = self.selu(self.dec_fc(x.view(-1, self.latent_dim)))
        
#         print(self.batch_size,
#                 self.out_channels_2,
#                 self.input_size_fc,
#                 self.input_size_fc)
        
        x = self.unpool(
            x.view(
                self.batch_size,
                self.out_channels_2,
                self.input_size_fc,
                self.input_size_fc
                ),
            self.indices_2
        )

        x = self.selu(self.conv_t1(x))

        x = self.unpool(
            x,
            self.indices_1
        )

        
        x = self.sigmoid(self.conv_t2(x))

        return x

    def forward(self, x): 
        x = self.encode(x)
        x = self.decode(x)

        return x 

    def calculate_output_featmap_size(self, in_size, kernel_size, padding, stride): 
        """
        Helper function to calculate size of the output 
        of a conv layer for square images. 
        """
        return int(np.ceil(in_size - kernel_size + 2*padding + stride) / stride) 



class AE(nn.Module): 

    def __init__(self, input_size, hidden_size, latent_dim): 
        super(AE, self).__init__()

        self.relu = nn.SELU()
        #self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(hidden_size)


        # Encoder 
        self.en_1 = nn.Linear(input_size, hidden_size)
        self.en_2 = nn.Linear(hidden_size, latent_dim)

        # Decoder 
        self.de_1 = nn.Linear(latent_dim, hidden_size)
        self.de_2 = nn.Linear(hidden_size, input_size)


    def encoder(self, x): 

        x = self.relu(self.batch_norm(self.en_1(x)))
        #x = self.relu(self.en_1(x))
        x = self.relu(self.en_2(x))

        return x 

    def decoder(self, x): 

        x = self.relu(
                self.batch_norm(
                    self.de_1(x)
                )
            )

        #x = self.relu(self.de_1(x))

        x = self.sigmoid(self.de_2(x))

        return x

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x 




class MLP(nn.Module):
    
    "Module for an MLP using one hidden layer for multiclass classification."


    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
        
        super(MLP, self).__init__()

        # During training, dropout zeroes out some of the elements
        # in the input tensor with prob= dropoput.
        self.dropout = nn.Dropout(p = dropout)

        self.batch_norm_hidden = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.log_softmax= nn.LogSoftmax(dim = 1)
        #self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.input2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)


    def encoder(self, input): 
        x = self.relu(self.batch_norm_hidden(self.input2hidden(input)))
        return x 


    def forward(self, input): 

        x = self.relu(self.batch_norm_hidden(self.input2hidden(input)))
        x = self.log_softmax(self.hidden2out(x))

        return x 

        

class BnLinear(nn.Module): 

    def __init__(self, input_dim, output_dim): 
        super(BnLinear, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x): 
        x = self.linear(x)
        x = self.bn(x)
        
        return x


class supervised_model(nn.Module): 
    """
    Deep multi-layer perceptron (MLP) for classification and regression.
    It is built with Linear layers that use Batch Normalization
    and tanh as activation functions. The model type is defined
    using the `model` argument. 
    
    Params
    ------
    dims (list): 
        Dimensions of the MLP. First element is the input dimension, 
        final element is the output dimension, intermediate numbers
        are the dimension of the hidden layers.

    model (str, default = 'regression'): 
        Type of supervised model. Options are 
        'regression': For MLP regression.
        'multiclass': For multiclass classification (single categorical variable).
        'binary': For binary classification.
        'multilabel': For multilabel classification, i.e when multiple 
        categorical columns are to be predicted.

        Notes: 'multiclass' uses F.log_softmax as activation 
        layer. Use nn.NLLLoss() as loss function.

    dropout (bool, default = True)

    """

    def __init__(self, dims, model = 'regression', dropout = True):
        

        super(supervised_model, self).__init__()

        self.output_dim = dims[-1]

        # Start range from 1 so that dims[i-1] = dims[0]
        linear_layers = [BnLinear(dims[i-1], dims[i]) for i in range(1, len(dims[:-1]))]

        self.fc_layers = nn.ModuleList(linear_layers)

        self.final_layer = BnLinear(dims[-2], self.output_dim)

        self.model = model
        self.dropout=dropout
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.cuda = torch.cuda.is_available()

    def project(self, x):
        "Projects data up to last hidden layer for visualization."

        for fc_layer in self.fc_layers[:-1]:
            x = fc_layer(x)
            x = self.tanh(x)

        x = self.fc_layers[-1](x)

        return x

    def project_to_latent_space(self, data_loader, n_feats, latent_dim):
        """
        Returns a generator to project dataset into latent space, 
        i.e. last hidden layer. 
        
        Params 
        ------
        data_loader (torch.DataLoader)
            DataLoader which handles the batches and parallelization. 
            
        n_feats (int)
            Number of dimensions of original dataset.
        
        latent_dim (int)
            Number of dimensions of layer to project onto.
            
        Returns (yields)
        -------
        encoded_sample (array-like generator)
            Generator of a single encoded data point in a numpy array format. 
        """
        
        # Set no_grad mode to avoid updating computational graph. 
        #with torch.no_grad()

        cuda = torch.cuda.is_available()
    

        # Iterate through all of the batches in the DataLoader
        for batch_x, targets in tqdm.tqdm(data_loader): 

            if cuda: 
                batch_x, targets = batch_x.cuda(), targets.cuda()
            
            # Reshape to eliminate batch dimension 
            batch_x = batch_x.view(-1, n_feats)
            
            # Project into latent space and convert tensor to numpy array
            if cuda: 
                batch_x_preds = self.project(batch_x.float()).cpu().detach().numpy()
            else: 
                batch_x_preds = self.project(batch_x.float()).detach().numpy()
            
            # For each sample decoded yield the line reshaped
            # to only a single array of size (latent_dim)
            for x in batch_x_preds:
                encoded_sample = x.reshape(latent_dim)
                
                yield encoded_sample


    def forward(self, x):

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = self.tanh(x)

        # Pass through final linear layer 
        if self.model == 'regression':  

            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            return x

        elif self.model == 'multiclass':
            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            x = F.log_softmax(x, dim = 1)

            return x

        elif self.model == 'binary':
            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            x = F.sigmoid(x)

            return x

        elif self.model == 'multilabel':
            if self.dropout:
                x = F.dropout(x)
            x = self.final_layer(x)
            x = F.sigmoid(x)

            return x

        else:
            print(self.model, ' is not a valid model type.')



class deep_linear_model(supervised_model):
    def __init__(self, dims, model = 'regression', dropout = True):
        super(deep_linear_model, self).__init__(dims)
        
        self.output_dim = dims[-1]

        # Start range from 1 so that dims[i-1] = dims[0]
        linear_layers = [BnLinear(dims[i-1], dims[i]) for i in range(1, len(dims[:-1]))]

        self.fc_layers = nn.ModuleList(linear_layers)

        self.final_layer = BnLinear(dims[-2], self.output_dim)

        self.model = model
        self.dropout=dropout
        
    def project(self, x):
        "Projects data up to last hidden layer for visualization."

        for fc_layer in self.fc_layers[:-1]:
            x = fc_layer(x)
            #x = self.tanh(x)

        x = self.fc_layers[-1](x)

        return x

    def forward(self, x): 

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            
        # Pass through final linear layer 
        if self.model == 'regression':  
            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            return x

        elif self.model == 'multiclass':
            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            x = F.log_softmax(x, dim = 1)
            return x

        elif self.model == 'binary':
            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            x = F.sigmoid(x)
            return x

        elif self.model == 'multilabel':
            if self.dropout:
                x = F.dropout(x)
            x = self.final_layer(x)
            x = F.sigmoid(x)
            return x

        else:
            print(self.model, ' is not a valid model type.')




class mlp_data_df(Dataset): 
    
    def __init__(self, df, nfeats, target_col_name, transform = False):
        """
        Generates data and target pairs for classification from a dataframe. 
        This version is suited for MLP problems.

        The assumption is that the numerical features are the first columns in the dataset.
        
        Note that this is a quick and dirty way to make torch.Datasets, but it is an order 
        of magnitude slower than just using numpy arrays as input. In this sense this object 
        is intended for use only on small dataframes. 
            
        Params
        ------
        df (pd.DataFrame)

        nfeats (int)
            Number of numerical features, numerical values have to be the first columns in the dataset.
            For extracting data the df gets sliced as df.iloc[ix, :nfeats]

        target_col_name(str)
            Name of the target column.

        transform(torch.transform)
            Optional kwarg to add a transformation to the dataset before returning the torch.tensor 

        """
        self.df = df
        self.transform = transform
        self.input_dim = nfeats
        self.target_col_name = target_col_name
        
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, ix): 
        
        if type(ix) == torch.Tensor: 
            ix = ix.tolist()
            
        # Get single row to process
        data = self.df.iloc[ix, :self.input_dim].values.astype(np.float64)
        target = self.df.iloc[ix][self.target_col_name]
        

        return data, target



class image_data_flatten_popAE(Dataset): 
    
    """
    Image dataset class for running the population autoencoder in torch. 

    Params 
    ------
    df (pd.DataFrame)
        Annotated dataframe containing the flattened images. 

    res (int)
        Resolution, i.e. length of each side of the image. 

    transform (torch.vision.transform, default=False)
        An auxiliary transformation like cropping or flipping. 

    conv (bool, default=False)
        Set to True if running a convolutional autoencoder to
         reshape the images.
    
    Returns
    -------
    image (torch.Tensor)
        A single image in tensor format. 

    """
    def __init__(self, df, res, transform = False, conv = False):
        self.transform = transform
        self.df = df
        self.res = res
        self.conv = conv
        
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, ix): 
        
        if type(ix) == torch.Tensor: 
            ix = ix.tolist()
            
        # Get image data
        image = self.df.iloc[ix, :int(self.res**2)].values.astype(np.float64)
                 
        if self.conv:
            image = image.reshape(1, self.res, self.res)
        
        if self.transform is not False: 
            image = self.transform(image)
        
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        return torch.from_numpy(data)



class unsupervised_dataset(Dataset): 
    
    """
    In-memory unsupervised dataset. 

    Params 
    ------
    path_to_data (str)
        Path to the dataset. 

    data (np.array)
        Numpy array containing expression data.

    Returns
    -------
    data_point(torch.Tensor)
        A single row of the dataset in torch.tensor fmt. 


    """
    def __init__(self, path_to_data, data= None, transform = False, conv = False):
        self.transform = transform

        if path_to_data is not None: 

            self.data = np.loadtxt(path_to_data, delimiter = '\t')
        
        elif data is not None: 
            self.data = data

        else: 
            print('Need to provide either a dataset in np.array fmt or a path to the file. ')
        
        #self.conv = conv
        #self.res = res
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, ix): 
        
        if type(ix) == torch.Tensor: 
            ix = ix.tolist()
            
        # Get a single row of matrix 
        data_point = self.data[ix, :].astype(np.float64)
                 
        # if self.conv:
        #     image = image.reshape(1, self.res, self.res)
        
        if self.transform is not False: 
            data_point = self.transform(data_point)
        
        return data_point



class adata_torch_dataset(Dataset): 
    """
    Base class for a single cell dataset in .h5ad, i.e. AnnData format
    This object enables building models in pytorch.
    It currently supports unsupervised (matrix factorization / autoencoder)
    and general supervised (classification/regression) models.

    Params
    ------
    data (ad.AnnData)
        AnnDataset containing the count matrix in the data.X object. 

    transform (torchvision.transforms, default= False)
        A torchvision.transforms-type transformation, e.g. ToTensor()

    supervised (bool, default = False)
        Indicator variable for supervised models. 

    target_col (string/array-like, default = None)
        If running a supervised model, target_col should be a column 
        or set of columns in the adata.obs dataframe. 
        When running a binary or multiclass classifier, the labels 
        should be in a single column in a int64 format. 
        I repeat, even if running a multiclass classifier, do not specify
        the columns as one-hot encoded. The one-hot encoded vector 
        will be specified in the classifier model. The reason is that, 
        nn.CrossEntropyLoss() and the more numerically stable nn.NLLLoss()
        takes the true labels as input in integer form (e.g. 1,2,3),
        not in one-hot encoded version (e.g. [1, 0, 0], [0, 1, 0], [0, 0, 1]).

        When running a multilabel classifier (multiple categorical columns,
        e.g ´cell_type´ and `behavior`), specify the columns as a **list**.

        In this case, we will use the nn.BCELoss() using the one-hot encoded 
        labels. This is akin to a multi-output classification.

    multilabel (bool, default = False)
        Indicator variable to specify a multilabel classifier dataset. 

    Returns
    -------
    data_point(torch.tensor)
        A single datapoint (row) of the dataset in torch.tensor format. 

    target(torch.tensor)
        If running supervised model, the "y" or target label to be predicted.
    """

    def __init__(
        self, data= None, transform = False, supervised = False,
        target_col = None, multilabel = False)->torch.tensor:

        self.data = data # This is the h5ad / AnnData 

        self.supervised = supervised 
        self.target_col = target_col
        self.transform = transform

        from scipy import sparse
        # Indicator of data being in sparse matrix format. 
        self.sparse = sparse.isspmatrix(data.X)

        self.multilabel = multilabel

        if self.multilabel:
            from sklearn.preprocessing import OneHotEncoder
            # Initialize one hot encoder
            enc = OneHotEncoder(sparse = False)
            self.one_hot_encoder = enc

            n_categories = len(self.target_col)

            # Extract target data 
            y_data = self.data.obs[self.target_col].values.astype(str).reshape(-1, n_categories)

            # Build one hot encoder
            self.one_hot_encoder.fit(y_data)

            # Get one-hot matrix and save as attribute
            self.multilabel_codes = self.one_hot_encoder.transform(y_data)
        
    def __len__(self):
        return self.data.n_obs
    
    def __getitem__(self, ix): 
        
        if type(ix) == torch.Tensor:
            ix = ix.tolist()
            
        # Get a single row of dataset and convert to numpy array if needed
        if self.sparse: 
            data_point = self.data[ix, :].X.A.astype(np.float64)
            
        else: 
            data_point = self.data[ix, :].X.astype(np.float64)
                 
        # if self.conv:
        #     image = image.reshape(1, self.res, self.res)

        if self.transform is not False: 
            data_point = self.transform(data_point)
        
        # Get all columns for multilabel classification codes
        if self.supervised and self.multilabel:
            target = self.multilabel_codes[ix, :]
            #target = self.transform(target)
            return data_point, target 

        # Get categorical labels for multiclass or binary classification 
        # or single column for regression (haven't implemented multioutput reg.)
        elif self.supervised:
            target  = self.data.obs.iloc[ix][self.target_col]
            #target = self.transform(target)
            return data_point, target

        # Fallback to unsupervised case.
        else:
            return data_point

    def codes_to_cat_labels(self, one_hot_labels): 
        """
        Returns categorical classes from labels in one-hot format.

        Params
        ------
        one_hot_labels (array-like)
            Labels of (a potentially new or predicted) dataset
            in one-hot-encoded format. 
        
        Returns 
        -------
        cat_labels(array-like, or list of array-like)
            Categorical labels of the one-hot encoded input. 

        """

        cat_labels = self.one_hot_encoder.inverse_transform(one_hot_labels)

        return cat_labels


class unsupervised_image_data(Dataset): 
    """
    Uses sklearn to load an image collection in memory. 


    Params 
    -------
    path (str)
        Path and pattern to images. Example = '../images/*.jpg'
    """

    def __init__(self, path, data = None, transform = None, conv =False):

        import skimage.io as sio 


        if path is not None: 
            self.data = sio.ImageCollection(path)

        elif data is not None: 
            self.data = data

        else: 
            print('Need to provide image dataset or path to images.')


        self.conv = conv
        self.transform = transform 

    def __len__(self): 
        return len(self.data)


    def __getitem__(self, ix): 

        if isinstance(ix, torch.Tensor):
            ix = ix.tolist()




        # Get only red channel 
        try: 
            image = self.data[ix][:, :, 0]
        except:
            image = self.data[ix]


        # Get all images to the same output size 
        # if self.resize: 
        #     from skimage.transform import resize

        #     output_shape = (200,200)
        #     image = resize(image, output_shape, anti_aliasing = True)
        
        if not self.conv: 
            
            image = image.flatten()

        if self.transform is not None: 
            image = self.transform(image)

        return image




class RNN_GRU(nn.Module): 
    """
    Generative or multi-class classifier RNN using the GRU cell.
    
    Params 
    ------
    input_size (int):
        Vocabulary size for initializing the embedding layer.
        
    embedding_size (int):
        Dimensionality of the embedding layer. 
    
    hidden_size (int):
        Dimensionality of the hidden state of the GRU. 
        
    output_size (int): 
        Dimensionality of the output. If it's in the generative mode, 
        it is the same as the input size. 
    
    input (torch.LongTensor):
        Sequence of indices.
    
    Returns
    -------
    output (torch.tensor): 
        Sequence of predictions. In the generative case it is a probability
        distribution over tokens. 
    
    last_hidden(torch.tensor): 
        Last hidden state of the GRU. 
        
    """
    
    def __init__(self, input_size, embedding_size, hidden_size,
                 output_size, n_layers = 1):
        
        super(RNN_GRU, self).__init__()
        
        self.input_size = input_size 
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_layers = n_layers 
        self.log_softmax = nn.LogSoftmax(dim= 1)
        
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        self.gru = nn.GRU(
            input_size = embedding_size,
            hidden_size= hidden_size,
            num_layers = n_layers
        )
    
        self.decoder = nn.Linear(hidden_size, output_size)

        
    def token_embedding(self, input): 
        
        "Helper function to map single character to the embedding space."
        
        projected = self.embedding(input)
        return projected 
    
    def forward(self, input): 
        
        embedding_tensor = self.embedding(input)
        
        # shape(seq_len = len(sequence), batch_size = 1, input_size = -1)
        embedding_tensor = embedding_tensor.view(len(input), 1, self.embedding_size)

        sequence_of_hiddens, last_hidden = self.gru(embedding_tensor)
        output_rnn = sequence_of_hiddens.view(len(input), self.hidden_size)

        output = self.decoder(output_rnn)
        # LogSoftmax the output for numerical stability
        output = self.log_softmax(output)
        return output, last_hidden


def train_rnn(rnn_model, input_tensor, target_tensor, loss_fn,
             optimizer, clip_grads = False, grad_clip_val= None):
    "Helper function to make a forward pass through a RNN model."
    
    # Zero out gradients 
    rnn_model.zero_grad()
    loss = 0
    
    # Make forward computation
    output, hidden = rnn_model(input_tensor)

    loss += loss_fn(output, target_tensor)
    
    # Backprop error and update params
    loss.backward()

     # Perform gradient clipping to avoid exploding/vanishing grads
    if clip_grads and grad_clip_val is not None:
        nn.utils.clip_grad_norm_(rnn_model.parameters(), grad_clip_val)

    elif clip_grads and grad_clip_val is None: 
        nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
    else: 
        pass


    optimizer.step()
    
    # Return average loss
    return loss.item() / len(input_tensor)



def train_vae(model, input_tensor, opt, batch_size, loss_fn = None): 

    """
    Wrapper function to make a forward pass through a VAE model.
    Assumes that the model class contains a variational loss function, 
    if this is not the case it can be provided using the loss_fn arg. 
    
    Params 
    ------
    model (torch.nn.Module)
        VAE model. 

    input_tensor (torch.tensor)
        Tensor of size (batch_size, n_feats). 

    opt (torch.optim.Optimizer)

    batch_size (int)

    loss_fn (default= None)
        Loss function in case VAE class doesn't contain a loss function. 


    Returns 
    -------

    loss (float)
        Variational loss for the input batch. 

    """

    input_ = input_tensor.view(batch_size, -1 ).float()
    
    # Zero out grads 
    opt.zero_grad()

    # Make forward computation 
    reconstructed, mu, log_var = model(input_)

    # Backprop errors 
    loss = model.loss(reconstructed, input_, mu, log_var)
    loss.backward()

    # Update weights
    opt.step()

    return loss




def tokenize_molecule(chemical_string, multiple_strings = True): 
    """
    Returns a list of single aminoacid characters from a sequence 
    of a chemical string, specifically SMILEs of SELFIES.
    

    Params 
    ------

    chemical_string (str): 
        A molecules in string format. 

    Returns
    -------
    tokens 


    Inspired by work from Xinhao Li : https://github.com/XinhaoLi74/SmilesPE

    """

    # Match special atoms inside brackets
    # it specifies matching any symbol that is not a backslash
    inside_brackets = "\[[^\]]+]"

    
    individual_atoms = "|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"

    individual_symbols = "\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$"

    multiple_closure_rings = "|\%[0-9]{2}|"

    numbers = "[0-9]"

    pattern =  inside_brackets + individual_atoms + individual_symbols\
              + multiple_closure_rings + numbers 


    regex = re.compile(pattern)
     
    tokens = [token for token in regex.findall(chemical_string)]


    return tokens 






class ChemVocab:
    
    def __init__(self, list_of_chemical_strings, min_freq = 0, tokenized = False,
                 flattened= False, other_tokens = None):
        """
        Generates a vocabulary object from a list of chemical strings(sequences), 
        such as SMILES or SELFIES. 
        
        Params 
        ------
        list_of_chemical_strings(array-like)
            Corpus of chemical strings formatted as a list of chemical strings.
        
        min_freq(int, default = 0)
            Minimum frequency for token appearance. Tokens with less counts than 
            this cutoff will be ignored. 
            
        tokenized (bool, default = False)
            If set to True, skips the tokenization function. 
            
        other_tokens (array-like, default = None)
            Optional arg for adding rare chemicals not in the following list: 
            [Br, Cl, N, O, S, P, F, I, B, C] 
        """


        if tokenized ==False:
            tokenized_sequences = [nm.atom_tokenizer(mol) for mol in list_of_chemical_strings]
        else: 
            tokenized_sequences = list_of_chemical_strings
        

        #Initialize dictionary of counts 
        counter = self.count_corpus(tokenized_sequences)
        
        # Sort tokens by decreasing counts
        self.token_freqs = sorted(counter.items(), key = lambda x:x[1], reverse = True)
        
        # Add a value for unknown tokens 
        self.unk = 0

        if other_tokens is None: 
            other_tokens = []

        uniq_tokens = ['<unk>'] + other_tokens

        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]

        self.n_tokens = len(uniq_tokens)

        self.token_to_ix = dict(zip(uniq_tokens, np.arange(0, self.n_tokens)))
        self.ix_to_token = dict(zip(np.arange(0, self.n_tokens), uniq_tokens))


    def __len__(self): 
        return self.n_tokens

    def __getitem__(self, tokens): 
        # If a single token, return token index 
        if not isinstance(tokens, (list, tuple)): 
            return self.token_to_ix.get(tokens, self.unk) 

        # Else get a list of token indices
        else: 
            #print('using __getitem__')
            return [self.token_to_ix.get(token, self.unk) for token in tokens]

    def to_tokens(self, indices): 
        # Get a single token from index
        if not isinstance(indices, (list, tuple)): 
            return self.ix_to_token[indices]

        return [self.ix_to_token[ix] for ix in indices]

    def count_corpus(self, sequences):  
        """
        Returns dictionary with counts per token from a list of list of tokens.
        """
        tokens = [tk for line in sequences for tk in line]
        return collections.Counter(tokens)


class mosesVAE(nn.Module):
    def __init__(self, vocab, config):

        """
        Recurrent VAE from the Moses project. 

        Params
        ------
        vocab (moses.CharVocab)
            Vocabulary for the chemical strings. 

        config (dict)
            Dictionary of arguments. Config possible keys : 
            embedding_size, hidden_size_enc, hidden_size_dec, latent_size, encoder_n_layers, 
            dropout, bidirectional, 
        
        """

        super().__init__()

        self.vocabulary = vocab
        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        # Word embeddings layer
        n_vocab = len(vocab)
        

        # Dict config 
        if 'freeze_embeddings' not in config.keys(): 
            config['freeze_embeddings'] = False


        if config['freeze_embeddings'] == True:
            self.x_emb.weight.requires_grad = False


        embedding_size = (64 if 'embedding_size' not in config.keys() else config['embedding_size'])
        hidden_size_enc = (256 if 'hidden_size_enc' not in config.keys() else config['hidden_size_enc'])
        hidden_size_dec = (512 if 'hidden_size_dec' not in config.keys() else config['hidden_size_dec'])
        latent_size = (128 if 'latent_size' not in config.keys() else config['latent_size'])
        q_n_layers = (1 if 'encoder_n_layers' not in config.keys() else config['encoder_n_layers'])
        q_dropout = (0.5 if 'dropout' not in config.keys() else config['dropout'])
        q_bidir = (False if 'bidirectional' not in config.keys() else config['bidirectional'])
        d_n_layers = (1 if 'decoder_n_layers' not in config.keys() else config['decoder_n_layers'])
        d_dropout = q_dropout


        # Word embedding layer
        self.x_emb = nn.Embedding(n_vocab, embedding_size, self.pad)
        
        # Encoder
        self.encoder_rnn = nn.GRU(
            embedding_size,
            hidden_size_enc,
            num_layers=q_n_layers,
            batch_first=True,
            dropout=q_dropout if q_n_layers > 1 else 0,
            bidirectional=q_bidir
        )


        q_d_last = hidden_size_enc * (2 if q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, latent_size)
        self.q_logvar = nn.Linear(q_d_last, latent_size)

        # Decoder
        
        self.decoder_rnn = nn.GRU(
            embedding_size + latent_size,
            hidden_size_dec,
            num_layers=d_n_layers,
            batch_first=True,
            dropout= d_dropout if d_n_layers > 1 else 0
        )


        self.decoder_lat = nn.Linear(latent_size, hidden_size_dec)
        self.decoder_fc = nn.Linear(hidden_size_dec, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)

        return kl_loss, recon_loss

    def forward_encoder(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss

    def forward_decoder(self, x, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )

        return recon_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=self.x_emb.weight.device)

    def sample(self, n_batch, max_len=100, z=None, temp=1.0):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)

            eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]



class mosesDataset:
    def __init__(self, vocab, data):
        """
        Creates a convenient Dataset with SMILES tokinization
        Arguments:
            vocab: CharVocab instance for tokenization
            data (list): SMILES strings for the dataset
        """
        self.vocab = vocab
        self.tokens = [vocab.string2ids(s) for s in data]
        self.data = data
        self.bos = vocab.bos
        self.eos = vocab.eos

    def __len__(self):
        """
        Computes a number of objects in the dataset
        """
        return len(self.tokens)

    def __getitem__(self, index):
        """
        Prepares torch tensors with a given SMILES.
        Arguments:
            index (int): index of SMILES in the original dataset
        Returns:
            A tuple (with_bos, with_eos, smiles), where
            * with_bos is a torch.long tensor of SMILES tokens with
                BOS (beginning of a sentence) token
            * with_eos is a torch.long tensor of SMILES tokens with
                EOS (end of a sentence) token
            * smiles is an original SMILES from the dataset
        """
        tokens = self.tokens[index]
        #with_bos = torch.tensor([self.bos] + tokens, dtype=torch.long)

        with_prefix = torch.tensor([self.bos] +tokens + [self.eos], dtype=torch.long)
        return with_prefix, self.data[index]

    def default_collate(self, batch, return_data=False):
        """
        Simple collate function for SMILES dataset. Joins a
        batch of objects from StringDataset into a batch
        Arguments:
            batch: list of objects from StringDataset
            pad: padding symbol, usually equals to vocab.pad
            return_data: if True, will return SMILES used in a batch
        Returns:
            with_bos, with_eos, lengths [, data] where
            * with_bos: padded sequence with BOS in the beginning
            * with_eos: padded sequence with EOS in the end
            * lengths: array with SMILES lengths in the batch
            * data: SMILES in the batch
        Note: output batch is sorted with respect to SMILES lengths in
            decreasing order, since this is a default format for torch
            RNN implementations
        """
        with_prefix, data = list(zip(*batch))
        lengths = [len(x) for x in with_prefix]

        order = np.argsort(lengths)[::-1]

        with_prefix = [with_prefix[i] for i in order]
        

        lengths = [lengths[i] for i in order]

        with_prefix = torch.nn.utils.rnn.pad_sequence(
            with_prefix, padding_value=self.vocab.pad
        )
        
        if return_data:
            data = np.array(data)[order]
            return with_prefix, lengths, data
        return with_prefix, lengths




def binary_cross_entropy(q,p): 
    """
    Returns the binary cross entropy of predictions (q) and real (p) binary vectors.
    
    Params
    ------
    q (torch.tensor)
        Binary vector of predicted values. 

    p (torch.tensor)
        Binary vector of real values. 
        
    Returns
    -------
    bce (torch.tensor)
        Binary cross entropy. 
    """
    bce = -torch.sum(p * torch.log(q + 1e-12) + (1 - p) * torch.log(1 - q + 1e-12), dim = -1)
    return bce




def one_hot_encode(cat_labels):

    """

    cat_labels (np.array): 
        Array with categorical labels. 
    
    """

    one_hot = np.zeros(shape = (cat_labels.shape[0], max(cat_labels)+ 1))

    for ix, val in enumerate(cat_labels): 
        one_hot[ix, val] = 1

    return one_hot

def multi_label_one_hot(cat_labels_list): 

    list_of_one_hot_mats = [one_hot_encode(labels) for labels in cat_labels_list]

    concat_hot = np.hstack(list_of_one_hot_mats)

    return concat_hot 



class supervised_dataset_clf(Dataset): 
    """
    
    The datasets are designed to be read in tsv format, 
    or fed in np.array or pd.DataFrame format. 

    This dataset is designed for classification. 

    Params 
    ------
    path_to_datasets(array-like)
        Tuple containing strings pointing to the datasets,
        i.e. : (path_to_x, path_to_y)

    data_x, data_y (array-like or pd.DataFrame)
        Datasets provided. data_y is designed to be an array 
        of categorical variables in string format. 
    
    multilabel (bool, default False)
        Set to True if it is a multilabel classification problem. 

    transform ()
    """

    def __init__(self, path_to_datasets = None, data_x = None, data_y= None,
                 multilabel = False, conv = False, transform_ = False): 

        
        self.transform_ = transform_
        
        if path_to_datasets is not None: 
            path_to_data_x, path_to_data_y = path_to_datasets
            self.data_x = np.loadtxt(path_to_data_x, delimiter = '\t')
            self.data_y_cat = np.loadtxt(path_to_data_y, delimiter = '\t')

        elif data_x is not None: 
            self.data_x = data_x
            self.data_y_cat = data_y 
        else: 
            print('Need to provide either a dataset in np.array fmt or a path to the file. ')
        
    
        self.multilabel = multilabel

        from sklearn.preprocessing import OneHotEncoder
        # One hot encode and get codes 
        enc = OneHotEncoder(sparse = False)
        self.one_hot_encoder = enc
        if self.multilabel :
            self.one_hot_encoder.fit(self.data_y_cat)
            codes = self.one_hot_encoder.transform(self.data_y_cat)
            
        else: 
            self.one_hot_encoder.fit(self.data_y_cat.reshape(-1, 1))

            codes = self.one_hot_encoder.transform(self.data_y_cat.reshape(-1, 1))

        # Save one-hot encoder inside Dataset object 
        
        self.data_y = codes
        #self.conv = conv
        #self.res = res

    def codes_to_cat_labels(self, one_hot_labels): 
        """
        Returns categorical classes from labels in one-hot format.

        Params
        ------
        one_hot_labels (array-like)
            Labels of (a potentially new or predicted) dataset
            in one-hot-encoded format. 
        
        Returns 
        -------
        cat_labels(array-like, or list of array-like)
            Categorical labels of the one-hot encoded input. 

        """

        cat_labels = self.one_hot_encoder.inverse_transform(one_hot_labels)

        return cat_labels


    def __len__(self): 
        try: 

            n_samples = self.data_x.shape[0]

        except: 
            n_samples = len(self.data_x)            

        return n_samples 


    def __getitem__(self, ix): 


        if type(ix)==torch.tensor: 
            ix = ix.tolist()


        data_point_x = self.data_x[ix, :].astype(np.float64)

        if self.multilabel: 
            data_point_y = self.data_y[ix, :].astype(np.int)
        else: 
            data_point_y = self.data_y[ix].astype(np.int)

        # if self.conv:
        #     data_point_x = data_point_x.reshape(1, self.res, self.res)
        
        if self.transform_ is not False: 
            data_point_x = self.transform_(data_point_x)
            data_point_y = self.transform_(data_point_y)

        return data_point_x, data_point_y



def train_supervised(
    model,
    input_tensor,
    y_true,
    loss_fn,
    optimizer,
    multiclass =False,
    n_out = 1,
    ):
    """
    Helper function to make forward and backward pass with minibatch.
    
    Params
    ------
    n_out (int, default = 1)
        Dimensionality of output dimension. Leave as 1 for multiclass, 
        i.e. the output is a probability distribution over classes (e.g. MNIST).
    """
    
    # Zero out grads 
    model.zero_grad()
    y_pred = model(input_tensor)
    
    #Note that if it's a multiclass classification (i.e. the output is a 
    # probability distribution over classes) the loss_fn 
    # nn.NLLLoss(y_pred, y_true) uses as input y_pred.size = (n_batch, n_classes)
    # and y_true.size = (n_batch), that's why it doesn't get reshaped. 

    if multiclass: 
        loss = loss_fn(y_pred, y_true)

    else: # Backprop error
        loss = loss_fn(y_pred, y_true.view(-1, n_out).float())
    
    loss.backward()
    # Update weights 
    optimizer.step()
    
    return loss

def validation_supervised(model, input_tensor, y_true, loss_fn, multiclass =False, n_classes= 1): 
    "Returns average loss for an input batch of data with a supervised model."
    y_pred = model(input_tensor.float())
    if multiclass:
        loss = loss_fn(y_pred, y_true)
    else:
        loss = loss_fn(y_pred, y_true.view(-1, n_classes).float())
    
    return loss.mean()


def supervised_trainer(
    n_epochs,
    train_loader, 
    val_loader, 
    model,
    criterion, 
    optimizer,
    multiclass = False, 
    n_classes = 1,
    train_prints_per_epoch = 5):
    """
    Helper function to train a supervised model for n_epochs. 
    
    Params
    ------
    n_classes (int, default = 1)
        Dimensionality of output dimension. Leave as 1 for multiclass, 
        i.e. the output is a probability distribution over classes (e.g. MNIST).
    """

    batch_size = train_loader.batch_size
    print_every = np.floor(train_loader.dataset.__len__() / batch_size / train_prints_per_epoch) # minibatches

    train_loss_vector = [] # to store training loss 
    val_loss_vector = np.empty(shape = n_epochs)

    cuda = torch.cuda.is_available()

    if cuda: 
        device = try_gpu()
        torch.cuda.set_device(device)
        model = model.to(device)

    for epoch in np.arange(n_epochs): 
        
        running_loss = 0

        # TRAINING LOOP
        for ix, (data, y_true) in enumerate(tqdm.tqdm(train_loader)): 
            
            input_tensor = data.view(batch_size, -1).float()
            
            if cuda: 
                input_tensor = input_tensor.cuda(device = device)
                y_true = y_true.cuda(device = device)
                
            train_loss = train_supervised(
                model,
                input_tensor, 
                y_true, 
                criterion, 
                optimizer,
                multiclass=multiclass,
                n_out =n_classes
                )
            
            running_loss += train_loss.item()
            
            # Print loss 
            if ix % print_every == print_every -1 : 
                
                # Print average loss 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, ix+1, running_loss / print_every))
                
                train_loss_vector.append(running_loss / print_every)
                
                # Reinitialize loss
                running_loss = 0.0
        
        # VALIDATION LOOP
        with torch.no_grad():
            validation_loss = []
            
            for i, (data, y_true) in enumerate(tqdm.tqdm(val_loader)):
                input_tensor = data.view(batch_size, -1).float()

                if cuda: 
                    input_tensor = input_tensor.cuda(device = device)
                    y_true = y_true.cuda(device = device)

                val_loss = validation_supervised(
                    model, input_tensor, y_true, criterion, multiclass, n_classes
                    )

                validation_loss.append(val_loss)
                
            mean_val_loss = torch.tensor(validation_loss).mean()
            val_loss_vector[epoch] = mean_val_loss
            
            print('Val. loss %.3f'% mean_val_loss)
        
    print('Finished training')

    return train_loss_vector, val_loss_vector


def supervised_model_predict(
    model,
    data_loader,
    criterion, 
    n_points = None, 
    n_feats= None,
    multiclass=False,
    n_outputs =1,
    score = True
    ):
    """
    Analog to model.predict_proba() from sklearn. Returns a prediction vector given a torch dataloder
    and model. It is designed for working with basic supervised models like binary or multilabel
    classification, and regression.

    Params
    ------

    model (torch.nn.model)
        Trained supervised model.

    data_loader

    n_points (int)
        Number of instances (rows) in the dataset. If not provided, the function will 
        try to extract it from the dataloader. 

    n_feats (int)
        Input dimensions for the model / number of columns in the dataset. If not provided,
        the function will try to extract it from the dataloader. 

    n_outputs (int, default = 1)
        Number of outputs of the model. Defaults to 1 dim output, for regression or 
        binary classification.

    Returns
    -------
    y_pred (np.array)
        Array with raw predictions from a forward pass of the model. 

    """
    if n_points == None and n_feats == None: 
        try:
            n_points, n_feats = data_loader.dataset.data.shape
        except: 
            print('Need to supply number of datapoints and features in input data.')

    batch_size = data_loader.batch_size

    cuda = torch.cuda.is_available()

    # Initialize predictions array 
    y_pred = torch.zeros(n_points, n_outputs)

    cum_sum = 0

    with torch.no_grad():

        for ix, (x, y) in tqdm.tqdm(enumerate(data_loader)):

            if cuda: 
                x, y = x.cuda(), y.cuda()

            # Reshape input for feeding to model 
            x = x.view(-1, n_feats)

            outputs = model(x.float())

            y_pred[ix * batch_size : ix * batch_size + batch_size, :] = outputs

            if score: 
                if multiclass:
                    if cuda: 
                        mean_loss = criterion(outputs, y).mean().cpu().detach().numpy()    
                    else: 
                        mean_loss = criterion(outputs, y).mean().detach().numpy()

                else:
                    if cuda:
                        mean_loss = criterion(outputs, y.view(-1, n_outputs).float()).mean().cpu().detach().numpy()
                    else: 
                        mean_loss = criterion(outputs, y.view(-1, n_outputs).float()).mean().detach().numpy()

                cum_sum+= mean_loss
                moving_average = cum_sum / (ix + 1)
        
        if score: 
            print("Mean validation loss: %.2f"%moving_average)

    return y_pred.detach().numpy()


def supervised_embeddings_numpy(
    X=None,
    y=None,
    get_predictions = True,
    regression = False,
    multiclass = False,
    batch_size = 16,
    model_type = 'binary',
    n_epochs = 5,
    learning_rate =1e-3,
    architecture_list = None,
    seed = 42
    )->np.ndarray:
    """
    Returns an array of embeddings using neural networks.

    The dataset X can be supplied in np.ndarray, pd.DataFrame or ad.AnnData
    format. The `y`

    # TO-DO: 
    # 1. Report accuracy on test set, project on all data points or give the option.
    # Currently only projecting on test set. 
    # 2. Early stopping. If the validation loss starts increasing, stop the training.
    # 3. Handle sparsity
    # 4. Handle stratifying by other variables in train-test split
    # 5. Handle multilabel and multioutput regression outputs. 

    Params
    ------
    X(np.ndarray, sparse.csr_matrix, or pd.DataFrame)
        Input dataset. It should

    y(array-like)
        If doing multiclass classification, input should be 1-d array
        of labels of dtype int. 
    
    model_type(str,defalt = 'binary')
        Type of supervised model to run. The options are 'binary',
        'multiclass', 'multilabel', 'regression'.
    

    Returns
    -------
    """
    # Extract array 
    if isinstance(X, pd.DataFrame):
        # Make AnnData
        adata = ad.AnnData(
            X = X.values,
            var = pd.DataFrame(columns = X.columns.values),
            obs = pd.DataFrame(y, columns =['label'] )
        )

    # TO-DO: make sparse if it saves up memory
    # if not sparse.isspmatrix_csr(X):
    #     X_sparse = sparse.csr_matrix(X)

    # sparsity = (1 - X_sparse.data.shape[0] / (X.shape[0]*X.shape[1]))*100

    # if sparsity >= 50:
    #     X = X_sparse

    else:
        # Make AnnData
        adata = ad.AnnData(
            X = X, 
            obs = pd.DataFrame(y, columns =['label'] )
        )

    # TO-DO :stratify by other variables
    from sklearn.model_selection import StratifiedShuffleSplit

    # Initialize stratified sampler
    splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3, random_state = seed)

    ixs = list(splitter.split(adata.X, adata.obs[['label']]))

    train_ix, val_ix = ixs[0][0], ixs[0][1]

    # Use numpy-like indices to split dataset 
    train_adata = adata[train_ix].copy()
    test_adata = adata[val_ix].copy()

    # Initialize torch dataset 
    train_dataset = adata_torch_dataset(
        train_adata, transform = transforms.ToTensor(), supervised = True, target_col = 'label'
    )

    test_dataset = adata_torch_dataset(
        test_adata, transform = transforms.ToTensor(), supervised = True, target_col = 'label'
    )

    # Get number of cores 
    n_cores = mp.n_cores

    # Initialize DataLoader for minibatching 
    train_loader = DataLoader(
        train_dataset, batch_size = batch_size, drop_last = True, shuffle = False, num_workers =n_cores
    )

    val_loader = DataLoader(
        test_dataset, batch_size = batch_size, drop_last = True, shuffle = False, num_workers = n_cores
    )

    # Only supporting single-output regression.
    if model_type == 'binary' or model_type == 'regression':
        n_cats = 1
    else:
        n_cats = int(adata.obs.label.unique().shape[0])

    #n_epochs = 5

    # Dimensionality of the layers in the neural network
    # we use 3 dimensions in the last hidden layer for visualization
    if architecture_list is None:
        dims = [adata.n_vars, 512, 128, 3, n_cats]
    else:
        dims = architecture_list

    model = supervised_model(dims, model = 'multiclass', dropout = False)
    model = initialize_network_weights(model, method = 'xavier_normal')

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0)


    if model_type =='binary' or model_type =='multilabel':
        criterion = nn.BCELoss()

    elif model_type == 'multiclass':
        criterion = nn.NLLLoss()
    
    elif model_type == 'regression':
        criterion = nn.MSELoss()
    else: 
        print('Need to provide a valid supervised model type.')

    print('Starting training')
    print('-------------------')
    train_loss, val_loss = supervised_trainer(
        n_epochs,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        multiclass = multiclass,
        n_classes = n_cats
    )

    print('Finished training.\n')
    

    proj_loader= DataLoader(
        test_dataset, batch_size = batch_size, drop_last = False, shuffle = False, num_workers = n_cores  
    )

    print('Starting predictions.')
    print('-------------------')
    with torch.no_grad():
        model.eval()
        y_hat = supervised_model_predict(
            model,
            proj_loader,
            criterion, 
            multiclass = multiclass,
            n_outputs = n_cats, 
            score = True
        )

    print('Finished predictions. \n')

    print('Starting embeddings.')
    print('---------------------')

    with torch.no_grad():
        model.eval()
        projection_arr = np.array(
            list(model.project_to_latent_space(proj_loader, dims[0], dims[-2]))
        )

    print('Finished embeddings.')

    df = pd.DataFrame(
       projection_arr, columns = ['latent_' + str(i) for i in range(1, dims[-2] + 1)]
    )


    # TO-DO: handle outputs from multilabel classifier. 
    # Add model predictions to df
    if multiclass:
        df['y_pred'] = y_hat.argmax(axis =1)
    elif multiclass ==False & model_type== 'binary':
        df['y_hat'] = y_hat
        df['y_pred'] = np.round(df['y_hat'])
    else:
        df['y_hat'] = y_hat

    df_proj = pd.concat([test_adata.obs, df.set_index(test_adata.obs.index)], axis=1)

    return df_proj, train_loss, val_loss, model

def supervised_embeddings_adata(
    adata, 
    label_col,
    train_ratio,
    return_embeddings_test_only = False,
    ):
    #TO-DO
    return None



# Resblock from https://github.com/calico/scnym/blob/master/scnym/model.py
class ResBlock(nn.Module):
    '''Residual block.
    References
    ----------
    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    arXiv:1512.03385
    '''

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
    ) -> None:
        '''Residual block with fully-connected neural network
        layers.
        Parameters
        ----------
        n_inputs : int
            number of input dimensions.
        n_hidden : int
            number of hidden dimensions in the Residual Block.
        Returns
        -------
        None.
        '''
        super(ResBlock, self).__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        # Build the initial projection layer
        self.linear00 = nn.Linear(self.n_inputs, self.n_hidden)
        self.norm00 = nn.BatchNorm1d(num_features=self.n_hidden)
        self.relu00 = nn.ReLU(inplace=True)

        # Map from the latent space to output space
        self.linear01 = nn.Linear(self.n_hidden, self.n_hidden)
        self.norm01 = nn.BatchNorm1d(num_features=self.n_hidden)
        self.relu01 = nn.ReLU(inplace=True)
        return

    def forward(self, x: torch.FloatTensor,
                ) -> torch.FloatTensor:
        '''Residual block forward pass.
        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, self.n_inputs]
        Returns
        -------
        o : torch.FloatTensor
            [Batch, self.n_hidden]
        '''
        identity = x

        # Project input to the latent space
        o = self.norm00(self.linear00(x))
        o = self.relu00(o)

        # Project from the latent space to output space
        o = self.norm01(self.linear01(o))

        # Make this a residual connection
        # by additive identity operation
        o += identity
        return self.relu01(o)

            
# class GraphConvolution(Module): 
    
#     """
#     Simple Graph Conv layer. 
#     """

#     def __init__(self, input_dim, output_dim):
#         super(GraphConvolution, self).__init__()
#         self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
#         self.bias = Parameter(torch.FloatTensor(output_dim))
#         self.reset_parameters()

#     def reset_parameters(self):
#         std_dev = 1./np.sqrt(self.weight.size(1))

#         self.weight.data.uniform_(-std_dev, std_dev)

#         self.bias.data.uniform_(-std_dev, std_dev)

#     def forward(self, input, adj): 

#         x = torch.mm(input, self.weight)
#         output = torch.spmm(adj, x)

#         return output + self.bias

# class BnGraphConvLayer(nn.Module): 

#     def __init__(self, input_dim, output_dim): 
#         super(BnGraphConvLayer, self).__init__()

#         self.graph_conv = GraphConvolution(input_dim, output_dim)
#         self.bn = nn.BatchNorm1d(output_dim)

#     def forward(self, input, adj): 
#         x = self.graph_conv(input_dim, adj)
#         x = self.bn(x)

#         return x 

# class GCN(nn.Module):

#     def __init__(self, input_dim, hidden_dim, n_classes):

#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(input_dim, hidden_dim)
#         self.gc2 = GraphConvolution(hidden_dim, n_classes)

#     def forward(self, x, adj): 
#         x = F.relu(self.gc1(x, adj))
#         x = self.gc2(x, adj)
#         out = F.log_softmax(x, dim = 1)
#         return out


# class DeepGCN(nn.Module): 

#     """
#     Deep Graph Convolutional Neural Network using batch normalization. 
#     """

#     def __init__(self, dims): 

#         """
#         Params 
#         ------
#         dims (list)
#             Designed to be supplied in the following format: 
#             [input_dim, [hidden_dims], output_dim]

#         """

#         super(DeepGCN, self).__init__()


#         [input_dim, h_dim, output_dim] = dims

#         neurons = [input_dim, *h_dim]

#         conv_layers = [BnGraphConvLayer(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

#         self.conv_layers = nn.ModuleList(conv_layers)
#         self.output_layer = BnGraphConvLayer(neurons[-1], output_dim)


#     def forward(self, x, adj): 

#         for GCN_layer in self.conv_layers: 
#             x = GCN_layer(x, adj)
#             x = F.relu(x)

#         x = self.output_layer(x, adj)
#         out = F.log_softmax(x, dim=1)

#         return out


