from cocoa_emu.dataset_generator import generate_dataset

generate_dataset(N=100000, 
                cobaya_yaml='projects/ROMAN_real_emu_emu/DC1_3x2_LCDM_PARAMS.yaml',
                sampling_mode='gaussian',
                use_mpi=True,
                T=512,
                covmat_file='projects/ROMAN_real_emu_emu/chains/dc1_3x2.covmat', 
                fiducial_params={'As_1e9': 2.5, 'ns': 0.97, 'H0': 71.0, 'omegab': 0.05,
                'omegam': 0.25, 'w0pwa': -1.0, 'w': -1.0, 'ROMAN_DZ_S1': 0.001414,
                'ROMAN_DZ_S2': 0.004298, 'ROMAN_DZ_S3': -0.002162, 'ROMAN_DZ_S4': 0.000047,
                'ROMAN_DZ_S5': 0.003450, 'ROMAN_DZ_S6': 0.002860, 'ROMAN_DZ_S7': 0.002578,
                'ROMAN_DZ_S8': -0.001002, 'ROMAN_A1_1': 0.606102, 'ROMAN_A1_2': -1.51541,
                'ROMAN_B1_1': 1.18, 'ROMAN_B1_2': 1.4, 'ROMAN_B1_3': 1.55, 'ROMAN_B1_4': 1.71,
                'ROMAN_B1_5': 1.9, 'ROMAN_B1_6': 2.15, 'ROMAN_B1_7': 2.52, 'ROMAN_B1_8': 3.44},
                output_dir='projects/ROMAN_real_emu_emu/chains/training_data_gaussian/',
                samples_filename='dc1_3x2_samples_9_22_512_lcdm.npy',
                datavectors_filename='dc1_3x2_datavectors_9_22_512_lcdm.npy',
                param_names_filename='dc1_3x2_param_names_9_22_512_lcdm.txt')