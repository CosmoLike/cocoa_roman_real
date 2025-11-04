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

def read_parameters(param_path, group_id):
	sample_path = os.path.join(param_path, f'group_{group_id}_parameters.npy')
	samples = np.load(sample_path)
	return samples


def generate_datavectors(samples, model, config, group_id, output_dir,use_mpi=True):
    """
    Generate data vectors for a group of samples, optionally using MPI.
    Each rank processes its share of samples, and results are gathered on rank 0.
    """
	temp_save_path = os.path.join(output_dir, 'temp', f'group_{group_id}')
	temp_save_file = os.path.join(temp_save_path, f'group_{group_id}_datavectors.npy')
	if os.path.exists(temp_save_file):
		print(f'group {group_id} already created')
		all_datavectors = np.load(temp_save_file)
		return all_datavectors
	
	else:
		print(f'Creating group {group_id} . . .')

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

    	n_samples = len(samples)
    	samples_per_rank = n_samples // size
    	remainder = n_samples % size

    	# Calculate start and end indices for this rank
    	start_idx = rank * samples_per_rank + min(rank, remainder)
    	end_idx = start_idx + samples_per_rank + (1 if rank < remainder else 0)

    	# Process samples assigned to this rank
    	local_datavectors = []
    	for i in range(start_idx, end_idx):
        	sample = samples[i]
        	param_dict = {p: float(sample[i]) for i, p in enumerate(config.running_params)}
        	param_dict['mnu']=0.06
        	param_dict['tau']=0.0697186
        	try:
            	dv = model.calculate_data_vector(param_dict)
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
            	temp_save_path = os.path.join(output_dir, 'temp', f'group_{group_id}')
            	os.makedirs(temp_save_path, exist_ok=True)
            	np.save(os.path.join(temp_save_path, f'group_{group_id}_datavectors.npy'), all_datavectors)
            	return all_datavectors
        	else:
            	return None
    	else:
        	# Non-MPI case
        	if len(local_datavectors) > 0:
            	temp_save_path = os.path.join(output_dir, 'temp', f'group_{group_id}')
            	os.makedirs(temp_save_path, exist_ok=True)
            	np.save(os.path.join(temp_save_path, f'group_{group_id}_datavectors.npy'), local_datavectors)
            	return local_datavectors
        	else:
            	raise ValueError(f"No valid data vectors were generated for group {group_id}!")

def generate_dataset(cobaya_yaml, use_mpi=True, 
                    sampling_mode='lhc', 
                    output_dir='./projects/roman_real/chains/training_data/',
                    samples_filename='parameter_samples.npy',
                    datavectors_filename='datavectors.npy',
                    param_names_filename='parameter_names.txt',
                    T=None, covmat_file=None, fiducial_params=None):
    """
    Resume generating dataset
    
    Parameters:
    -----------
    cobaya_yaml : str
        Path to cobaya configuration file
    use_mpi : bool, optional
        Whether to use MPI for parallel processing (default: True)
    output_dir : str, optional
        Directory to save output files (default: './projects/roman_real/chains/training_data/')
    samples_filename : str, optional
        Filename for parameter samples (default: 'parameter_samples.npy')
    datavectors_filename : str, optional
        Filename for data vectors (default: 'datavectors.npy')
    param_names_filename : str, optional
        Filename for parameter names (default: 'parameter_names.txt')
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
    
    config = cocoa_config(cobaya_yaml)
    model = CocoaModel(cobaya_yaml, config.likelihood)

	n_groups = 50
	sample_groups = []
	for i in range(n_groups):
		sample_groups[i] = read_parameters(temp_path, group_id)

    
    # Broadcast sample groups to all processes if using MPI
    if use_mpi:
        try:
            for i in range(50):
                sample_groups[i] = comm.bcast(sample_groups[i], root=0)
        except:
            if rank == 0:
                print("Warning: MPI broadcast failed, falling back to serial mode")
            use_mpi = False
    
    all_datavectors = []
    for group_id in range(50):
        if rank == 0:
            print(f'\nProcessing group {group_id}...')
        group_datavectors = generate_datavectors(sample_groups[group_id], model, config, group_id,  output_dir=output_dir, use_mpi=use_mpi)
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

