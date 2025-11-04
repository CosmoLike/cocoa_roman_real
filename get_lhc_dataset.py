from cocoa_emu.dataset_generator import generate_dataset

generate_dataset(N=1000000, 
                cobaya_yaml='projects/ROMAN_real_emu_emu/lhc_params_tight.yaml',
                sampling_mode='lhc',
                use_mpi=True,
                T=None,
                covmat_file='projects/ROMAN_real_emu_emu/dc1_3x2.covmat', 
                fiducial_params={'As_1e9': 2.5, 'ns': 0.97, 'H0': 71.0, 'omegab': 0.05,
                'omegam': 0.25, 'w0pwa': -1.0, 'w': -1.0, 'ROMAN_DZ_S1': 0.001414,
                'ROMAN_DZ_S2': 0.004298, 'ROMAN_DZ_S3': -0.002162, 'ROMAN_DZ_S4': 0.000047,
                'ROMAN_DZ_S5': 0.003450, 'ROMAN_DZ_S6': 0.002860, 'ROMAN_DZ_S7': 0.002578,
                'ROMAN_DZ_S8': -0.001002, 'ROMAN_A1_1': 0.606102, 'ROMAN_A1_2': -1.51541},
                output_dir='projects/ROMAN_real_emu_emu/chains/training_data/',
                samples_filename='dc1_3x2_samples_8_18_w0wa_tight.npy',
                datavectors_filename='dc1_3x2_datavectors_8_18_w0wa_tight.npy',
                param_names_filename='dc1_3x2_param_names_8_18_w0wa_tight.txt')