# Generate training dataset

import numpy as np
import emcee
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import sys
import os
from cocoa_emu import cocoa_config, CocoaModel
from scipy.stats import qmc

sys.path.append(os.path.dirname(__file__))


class generate_samples:
	def __init__(self, N, config=None, sampling_mode='lhc', T=None, covmat_file=None, fiducial_params=None):
		"""
		Initialize the sample generator
		
		Parameters:
		-----------
		N : int
			Number of samples to generate
		config : Config, optional
			Configuration object containing parameter ranges and priors
		sampling_mode : str, optional
			Sampling mode: 'lhc' or 'gaussian' (default: 'lhc')
		T : float, optional
			Temperature parameter for gaussian sampling (required for gaussian mode)
		covmat_file : str, optional
			Covariance matrix file for gaussian sampling (required for gaussian mode)
		fiducial_params : dict, optional
			Fiducial parameters for gaussian sampling (required for gaussian mode)
		"""
		self.N = int(N)  # Ensure N is an integer
		self.sampling_mode = sampling_mode
		self.config = config
		self.T = T
		self.covmat_file = covmat_file
		self.fiducial_params = fiducial_params
		
		# Validate gaussian mode parameters
		if sampling_mode == 'gaussian':
			if T is None or covmat_file is None or fiducial_params is None:
				raise ValueError("For gaussian sampling mode, T, covmat_file, and fiducial_params must be provided")
		
		self.sampled_params = config.running_params
		self.param_ranges = {}
		for param in self.sampled_params:
			prior_info = config.priors[param]
			if isinstance(prior_info, dict) and 'min' in prior_info and 'max' in prior_info:
				self.param_ranges[param] = (prior_info['min'], prior_info['max'])
			else:
				# If no explicit prior range, use a wide default range
				center = prior_info['loc']
				std = np.array(prior_info['scale'])
				self.param_ranges[param] = (center - 3.0*std, center + 3.0*std)

		self.sampling_dim = len(self.sampled_params)

	def _get_lhc_samples(self):
		"""
		Generate LHC samples for the parameters, discarding those that fail hard priors.
		"""
		sample_generator = qmc.LatinHypercube(d=self.sampling_dim)
		samples = sample_generator.random(self.N)
		l_bounds = np.array([self.param_ranges[p][0] for p in self.sampled_params])
		u_bounds = np.array([self.param_ranges[p][1] for p in self.sampled_params])
		scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
		
		return np.array(scaled_samples)

	def _get_samples(self):
		"""
		Generate samples based on the sampling mode
		"""
		if self.sampling_mode == 'lhc':
			return self._get_lhc_samples()
		elif self.sampling_mode == 'gaussian':
			return self.run_mcmc()
		else:
			raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

	def _read_parameter_names_from_covmat(self, covmatfile):
		"""
		Read parameter names from the first line of the covariance matrix file.
		Handles '# ' prefix that might be present.
		
		Parameters:
		-----------
		covmatfile : str
			Path to the covariance matrix file
			
		Returns:
		--------
		list
			List of parameter names in the order they appear in the covariance matrix
		"""
		try:
			with open(covmatfile, 'r') as f:
				first_line = f.readline().strip()
			
			# Remove '# ' prefix if present
			if first_line.startswith('# '):
				first_line = first_line[2:]  # Remove the '# ' prefix
			elif first_line.startswith('#'):
				first_line = first_line[1:]  # Remove just the '#' prefix
			
			# Split the first line by whitespace and clean up
			param_names = [name.strip() for name in first_line.split()]
			
			# Remove any empty strings
			param_names = [name for name in param_names if name]
			
			print(f"Read parameter names from covariance matrix file:")
			for i, name in enumerate(param_names):
				print(f"  {i}: {name}")
			
			return param_names
		except Exception as e:
			raise ValueError(f"Failed to read parameter names from covariance matrix file {covmatfile}: {str(e)}")

	def hard_priors(self, param, value):
		"""Check if parameters are within hard prior bounds"""
		min_val, max_val = self.param_ranges[param]
		if value < min_val or value > max_val:
			return -np.inf
		return 0.0

	def gaussian_prior(self, param, value):
		"""Calculate Gaussian prior for parameters"""
		ans = 0.0
		prior = self.config.priors[param]
		mu = prior["loc"]
		std = prior["scale"]
		y = (value - mu) / std
		ans += -0.5 * y * y
		return ans

	def derived_constraints(self, param_dict):
		ans = 0.0
		# Derived quantities constraints
		# Total matter density (baryons + CDM) should be reasonable
		if "omegam" in param_dict and "omegab" in param_dict:
			omegam = param_dict["omegam"]
			omegab = param_dict["omegab"]
			omegac = omegam - omegab  # CDM density
			if omegac < 0.05 or omegac > 0.8:  # CDM should be reasonable
				ans += -np.inf
		
		# BBN constraint (if both omegab and H0 are present)
		if "omegab" in param_dict and "H0" in param_dict:
			omegab = param_dict["omegab"]
			H0 = param_dict["H0"]
			ombh2 = omegab * (H0/100)**2
			if ombh2 < 0.005 or ombh2 > 0.04:
				ans += -np.inf

		#w0wa restrictions
		if "w" in param_dict and "wopwa" in param_dict:
			w0 = param_dict["w"]
			w0pwa = param_dict["wopwa"]
			if w0>= -0.01 or w0pwa>-0.01:
				ans += -np.inf
				
		return ans

	# setup likelihood
	def lnprior(self, param):
		"""Calculate log prior probability with hard priors for cosmo and IA"""
		ans = 0.0
		param_dict = {k:v for k,v in zip(self.sampled_params, param)}
		
		for i, par in enumerate(self.sampled_params):
			prior = self.config.priors[par]
			dist = prior.get("dist", "uniform")
			if dist == "uniform":
				ans += self.hard_priors(par, param_dict[par])
			elif dist == "norm":
				ans += self.gaussian_prior(par, param_dict[par])
		
		ans += self.derived_constraints(param_dict)
		
		return ans

	def lnlkl(self, param):
		"""Calculate log likelihood using Gaussian approximation with dzs sampling"""
		# Only use the sampled parameters for likelihood calculation
		param_dict = {k:v for k,v in zip(self.sampled_params, param)}
		
		# Read parameter names and covariance matrix if not already done
		if not hasattr(self, 'inv_cov') or not hasattr(self, 'full_param_list'):
			self.full_param_list = self._read_parameter_names_from_covmat(self.covmat_file)
			covmat = np.loadtxt(self.covmat_file, skiprows=1)
			self.inv_cov = np.linalg.inv(covmat)
			
			# Create mapping from sampled parameters to covariance matrix indices
			self.param_to_cov_index = {}
			for i, param in enumerate(self.full_param_list):
				if param in self.sampled_params:
					self.param_to_cov_index[param] = i
		
		# Create parameter array in the same order as covariance matrix
		param_array = np.zeros(len(self.full_param_list))
		fiducial_array = np.zeros(len(self.full_param_list))
		
		# Fill in values for parameters that exist in both sampled and covariance matrix
		for param in self.sampled_params:
			if param in self.param_to_cov_index:
				cov_index = self.param_to_cov_index[param]
				param_array[cov_index] = param_dict[param]
				fiducial_array[cov_index] = self.fiducial_params[param]
		
		diff = param_array - fiducial_array
		lkl = (-0.5/self.T) * (diff @ self.inv_cov @ diff)
		return lkl

	def lnpost(self, param):
		"""Calculate log posterior (prior + likelihood)"""
		lp = self.lnprior(param)
		ll = self.lnlkl(param)
		posterior = lp + ll
		return posterior

	def initialize_gaussian_samples(self,T,covmat_file,fiducial_params):
		"""
		Generate Gaussian samples for the parameters, discarding those that fail hard priors.
		"""
		if covmat_file is not None:
			# Read parameter names from the first line of the covariance matrix file
			full_param_list = self._read_parameter_names_from_covmat(covmat_file)
			
			# Load the covariance matrix (skip the header line)
			covmat = np.loadtxt(covmat_file, skiprows=1)
			inv_cov = np.linalg.inv(covmat)
			
			# Create mapping from sampled parameters to covariance matrix indices
			param_to_cov_index = {}
			for i, param in enumerate(full_param_list):
				if param in self.sampled_params:
					param_to_cov_index[param] = i
			
			print(f"Loaded covariance matrix with {len(full_param_list)} parameters")
			print(f"Mapped {len(param_to_cov_index)} sampled parameters to covariance indices")
		else:
			raise ValueError("Covariance matrix file is required for Gaussian sampling")

		# Create initialization spreads based on parameter type
		#init_spreads = [0.05, 0.001, 0.1, 0.001, 0.01,
		#0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
		#0.1, 0.1]
		#init_spreads = [0.05, 0.001, 0.1, 0.001, 0.01, 
		#0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
		#0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
		#0.1, 0.1]
		init_spreads = np.zeros(self.sampling_dim)
		param_std = np.diag(covmat)**0.5
		
		for i, param in enumerate(self.sampled_params):
			if param in param_to_cov_index:
				cov_index = param_to_cov_index[param]
				init_spreads[i] = param_std[cov_index]
				print(f"Parameter {param}: using covariance spread = {init_spreads[i]:.6f}")
			else:
				init_spreads[i] = 0.05
				print(f"Parameter {param}: using covariance spread = {init_spreads[i]:.6f}")
		"""	if param == "H0":
				init_spreads[i] = 2.0
			elif param in param_to_cov_index:
				cov_index = param_to_cov_index[param]
				
				init_spreads[i] = 3.0 * np.sqrt(covmat[cov_index, cov_index])
				print(f"Parameter {param}: using covariance spread = {init_spreads[i]:.6f}")
			else:
				init_spreads[i] = 0.005
				print(f"Parameter {param}: using prior-based spread = {init_spreads[i]:.6f}")"""
		
		# Create initial positions using fiducial values for sampled parameters
		#fiducial_values = np.array([fiducial_params.get(p, 0.0) for p in self.sampled_params])
		fiducial_values = np.array([fiducial_params[p] for p in self.sampled_params])
		
		# Calculate number of walkers needed for MCMC
		n_walkers = max(100, 4*self.sampling_dim)  # At least 100 walkers or 4x dimension
		
		# Create positions for all walkers, not just final samples
		pos = np.array(fiducial_values) + 0.3 * np.array(init_spreads) * np.random.normal(size=(n_walkers, self.sampling_dim))
		return pos	

	def run_mcmc(self, n_threads=1, use_mpi=True):
		"""
		Run the MCMC sampler for gaussian mode
		"""
		if self.sampling_mode != 'gaussian':
			raise ValueError("MCMC sampling is only available for gaussian mode")
		
		n_walkers = max(100, 4*self.sampling_dim)  # At least 100 walkers or 4x dimension
		n_steps = int(self.N//n_walkers + 100)  # Extra 100 steps for burn-in, ensure integer

		pos = self.initialize_gaussian_samples(self.T, self.covmat_file, self.fiducial_params)
		sampler = emcee.EnsembleSampler(n_walkers, self.sampling_dim, self.lnpost)
		sampler.run_mcmc(pos, n_steps, progress=True)
		
		# Discard burn-in and thin
		chain = sampler.get_chain()
		samples = sampler.get_chain(discard=50, thin=max(1, int(chain.shape[0]//self.N)))
		flat_samples = samples.reshape((-1, self.sampling_dim))
		
		# Ensure we have exactly N samples
		if len(flat_samples) > self.N:
			flat_samples = flat_samples[:int(self.N)]
		
		return flat_samples

	def get_sampling_info(self):
		"""Return information about the sampling approach"""
		info = {
			'total_parameters': len(self.sampled_params),
			'sampling_mode': self.sampling_mode,
			'number_of_samples': self.N
		}
		return info

	def print_sampling_info(self):
		"""Print detailed information about the sampling approach"""
		info = self.get_sampling_info()
		print("\n" + "="*60)
		print("SAMPLING CONFIGURATION")
		print("="*60)
		print(f"Total parameters: {info['total_parameters']}")
		print(f"Sampling mode: {info['sampling_mode']}")
		print(f"Number of samples: {info['number_of_samples']}")
		print(f"Parameter ranges:")
		for param in self.sampled_params:
			print(f"  - {param}: {self.param_ranges[param]}")
		
		print("="*60)

def generate_datavectors(samples, model, config, group_id, use_mpi=True, output_dir='./projects/roman_real/chains/training_data/'):
    """
    Generate data vectors for a group of samples, optionally using MPI.
    Each rank processes its share of samples, and results are gathered on rank 0.
    """
    if use_mpi:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        except ImportError:
            print("Warning: MPI not available, falling back to serial mode")
            rank = 0
            size = 1
            use_mpi = False
    else:
        rank = 0
        size = 1

    #print(size)
    n_samples = len(samples)
    samples_per_rank = n_samples // size
    remainder = n_samples % size

    #print(samples_per_rank)

    # Calculate start and end indices for this rank
    start_idx = rank * samples_per_rank + min(rank, remainder)
    end_idx = start_idx + samples_per_rank + (1 if rank < remainder else 0)

    # Process samples assigned to this rank
    local_datavectors = []
    for i in range(start_idx, end_idx):
        sample = samples[i]
        param_dict = {p: float(sample[i]) for i, p in enumerate(config.running_params)}
        #param_dict['w0pwa']=-1.0
        #param_dict['w']=-1.0
        param_dict['mnu']=0.06
        param_dict['tau']=0.0697186
        #print(f"Sample {i}, param_dict: {param_dict}")  # DEBUG: print parameters
        try:
            dv = model.calculate_data_vector(param_dict)
            #print(f"Sample {i}, data vector: {dv}")  # DEBUG: print data vector
            if dv is not None:
                if np.any(np.isnan(dv)):
                    print(f"Sample {i} produced NaNs in data vector!")
                local_datavectors.append(dv)
            else:
                print(f"Warning: No data vector generated for parameters: {param_dict}")
        except Exception as e:
            print(f"Warning: Failed to generate data vector for parameters: {param_dict}")
            print(f"Error: {str(e)}")
            continue

    local_datavectors = np.array(local_datavectors)

    if use_mpi:
        # Gather all data vectors to rank 0
        all_datavectors = comm.gather(local_datavectors, root=0)
        if rank == 0:
            # Concatenate results from all ranks
            all_datavectors = np.concatenate([dv for dv in all_datavectors if len(dv) > 0])
            temp_save_path = os.path.join(output_dir, 'temp2', f'group_{group_id}')
            os.makedirs(temp_save_path, exist_ok=True)
            np.save(os.path.join(temp_save_path, f'group_{group_id}_datavectors.npy'), all_datavectors)
            return all_datavectors
        else:
            return None
    else:
        # Non-MPI case
        if len(local_datavectors) > 0:
            temp_save_path = os.path.join(output_dir, 'temp2', f'group_{group_id}')
            os.makedirs(temp_save_path, exist_ok=True)
            np.save(os.path.join(temp_save_path, f'group_{group_id}_datavectors.npy'), local_datavectors)
            return local_datavectors
        else:
            raise ValueError(f"No valid data vectors were generated for group {group_id}!")

def generate_dataset(N, cobaya_yaml, use_mpi=True, 
                    sampling_mode='lhc', 
                    output_dir='./projects/roman_real/chains/training_data/',
                    samples_filename='parameter_samples.npy',
                    datavectors_filename='datavectors.npy',
                    param_names_filename='parameter_names.txt',
                    T=None, covmat_file=None, fiducial_params=None):
    """
    Generate a training dataset with hard priors for cosmology and IA parameters.
    
    Parameters:
    -----------
    N : int
        Number of samples
    cobaya_yaml : str
        Path to cobaya configuration file
    use_mpi : bool, optional
        Whether to use MPI for parallel processing (default: True)
    sampling_mode : str, optional
        Sampling mode: 'lhc' or 'gaussian' (default: 'lhc')
    output_dir : str, optional
        Directory to save output files (default: './projects/roman_real/chains/training_data/')
    samples_filename : str, optional
        Filename for parameter samples (default: 'parameter_samples.npy')
    datavectors_filename : str, optional
        Filename for data vectors (default: 'datavectors.npy')
    param_names_filename : str, optional
        Filename for parameter names (default: 'parameter_names.txt')
    T : float, optional
        Temperature parameter for gaussian sampling (required for gaussian mode)
    covmat_file : str, optional
        Covariance matrix file for gaussian sampling (required for gaussian mode)
    fiducial_params : dict, optional
        Fiducial parameters for gaussian sampling (required for gaussian mode)
    
    Features:
    ---------
    - Hard priors for cosmological parameters (omega_m, H0, ns, As_1e9, etc.)
    - Supports both Latin Hypercube (LHC) and Gaussian sampling modes
    """
    # Initialize MPI if available
    if use_mpi:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        except ImportError:
            rank = 0
    else:
        rank = 0
    
    # Create output directories
    temp_path = os.path.join(output_dir, 'temp')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)
    
    # Only rank 0 does the sampling
    if rank == 0:
        config = cocoa_config(cobaya_yaml)
        model = CocoaModel(cobaya_yaml, config.likelihood)
        
        print('Generating samples for parameters:')
        print(config.running_params)
        print("Running params:", config.running_params)
        print("All params:", config.priors.keys())
        
        if sampling_mode == 'gaussian':
            print('\nUsing Gaussian sampling mode')
            if T is None or covmat_file is None or fiducial_params is None:
                raise ValueError("For gaussian sampling mode, T, covmat_file, and fiducial_params must be provided")
        else:
            print('\nUsing LHC sampling mode')
        
        # Generate samples
        print(f"\nGenerating {sampling_mode.upper()} samples...")
        sampler = generate_samples(N, config=config, sampling_mode=sampling_mode, 
                                 T=T, covmat_file=covmat_file, fiducial_params=fiducial_params)
        
        # Print sampling configuration
        sampler.print_sampling_info()
        
        # Generate samples based on mode
        if sampling_mode == 'lhc':
            samples = sampler._get_lhc_samples()
        elif sampling_mode == 'gaussian':
            samples = sampler.run_mcmc(n_threads=1, use_mpi=False)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        
        # Save samples immediately after generation
        samples_path = os.path.join(output_dir, samples_filename)
        np.save(samples_path, samples)
        print(f"\nSaved {len(samples)} parameter samples to {samples_path}")
        
        # Split samples into 10 groups
        n_groups = 10
        samples_per_group = len(samples) // n_groups
        sample_groups = []
        for i in range(n_groups):
            start_idx = i * samples_per_group
            end_idx = start_idx + samples_per_group if i < n_groups - 1 else len(samples)
            sample_groups.append(samples[start_idx:end_idx])
            # Save each group's parameters
            group_filename = f'group_{i}_parameters.npy'
            np.save(os.path.join(temp_path, group_filename), sample_groups[-1])
    else:
        config = cocoa_config(cobaya_yaml)
        model = CocoaModel(cobaya_yaml, config.likelihood)
        sample_groups = [None] * 10  # Placeholder for non-root ranks
    
    # Broadcast sample groups to all processes if using MPI
    if use_mpi:
        try:
            for i in range(10):
                sample_groups[i] = comm.bcast(sample_groups[i], root=0)
        except:
            if rank == 0:
                print("Warning: MPI broadcast failed, falling back to serial mode")
            use_mpi = False
    
    # Generate data vectors for each group
    if rank == 0:
        print('\nGenerating data vectors for each group...')
    
    all_datavectors = []
    for group_id in range(10):
        if rank == 0:
            print(f'\nProcessing group {group_id}...')
        group_datavectors = generate_datavectors(sample_groups[group_id], model, config, group_id, use_mpi=use_mpi, output_dir=output_dir)
        if rank == 0 and group_datavectors is not None:
            all_datavectors.extend(group_datavectors)
    
    # Only rank 0 combines and saves the final results
    if rank == 0:
        all_datavectors = np.array(all_datavectors)
        datavectors_path = os.path.join(output_dir, datavectors_filename)
        np.save(datavectors_path, all_datavectors)
        
        # Save parameter names
        param_names_path = os.path.join(output_dir, param_names_filename)
        with open(param_names_path, 'w') as f:
            for param in config.running_params:
                f.write(param + '\n')
        
        print(f'\nFinal results:')
        print(f'Total samples: {len(samples)}')
        print(f'Total data vectors: {len(all_datavectors)}')
        print(f'Results saved to {output_dir}')
        print(f'  - Parameter samples: {samples_path}')
        print(f'  - Data vectors: {datavectors_path}')
        print(f'  - Parameter names: {param_names_path}')
        print(f'Temporary results saved to {temp_path}')
        return True
    return None

