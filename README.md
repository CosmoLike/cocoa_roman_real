## Running Cosmolike projects (Basic instructions) <a name="roman_running_cosmolike_projects"></a> 

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment the following lines on `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which cosmolike projects will be installed and compiled
>     # ------------------------------------------------------------------------------
>     #export IGNORE_COSMOLIKE_LSSTY1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     #export IGNORE_COSMOLIKE_ROMAN_FOURIER_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_REAL_CODE=1
>
>     (...)
> 
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export ROMAN_REAL_URL="https://github.com/CosmoLike/cocoa_roman_real.git"
>     export ROMAN_REAL_NAME="roman_real"
>     #BRANCH: if unset, load the latest commit on the specified branch
>     #export ROMAN_REAL_BRANCH="main"
>     #COMMIT: if unset, load the specified commit
>     export ROMAN_REAL_COMMIT="23a774c32480b7b4bd5da5f637270310bc88f86c"
>     #BRANCH: if unset, load the specified TAG
>     #export ROMAN_REAL_TAG="v4.0-beta17"


> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
> 
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
> 
>       source ./installation_scripts/compile_all_projects.sh
> 
> or (in case users just want to compile roman_real project)
>
>       source ./projects/roman_real/scripts/compile_roman_real.sh

> [!TIP]
> Assuming Cocoa is installed on a local (not remote!) machine, type the command below after step 2️⃣ to run Jupyter Notebooks.
>
>     jupyter notebook --no-browser --port=8888
>
> The terminal will then show a message similar to the following template:
>
>     (...)
>     [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
>     [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>     [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
>
> Now go to the local internet browser and type `http://127.0.0.1:8888/?token=XXX`, where XXX is the previously saved token displayed on the line
> 
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>
> The project roman_real contains jupyter notebook examples located at `projects/roman_real`.

To run the example

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment 
    
      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).
    
    export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE
      
 **Step :three:**: The folder `projects/roman_real` contains examples. So, run the `cobaya-run` on the first example following the commands below.

- **One model evaluation**:

  - Linux

        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/roman_real/EXAMPLE_EVALUATE1.yaml -f

  -  macOS (arm)

         mpirun -n 1 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/roman_real/EXAMPLE_MCMC1.yaml -f

   -  macOS (arm)
     
          mpirun -n 4 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_MCMC1.yaml -f

# Running ML emulators <a name="roman_examples_emul"></a>

Cocoa contains a few transformer- and CNN-based neural network emulators capable of simulating the CMB, cosmolike outputs, matter power spectrum, and distances. We provide a few scripts that exemplify their API. To run them, users ensure the following lines are commented out in `set_installation_options.sh` before running the `setup_cocoa.sh` and `compile_cocoa.sh`. By default, these lines should be commented out, but it is worth checking.

      [Adapted from Cocoa/set_installation_options.sh shell script] 
      # insert the # symbol (i.e., unset these environmental keys  on `set_installation_options.sh`)
      #export IGNORE_EMULTRF_CODE=1              #SaraivanovZhongZhu (SZZ) transformer/CNN-based emulators
      #export IGNORE_EMULTRF_DATA=1            
      #export IGNORE_LIPOP_LIKELIHOOD_CODE=1     # to run EXAMPLE_EMUL_(EVALUATE/MCMC/NAUTILUS/EMCEE1).yaml
      #export IGNORE_LIPOP_CMB_DATA=1           
      #export IGNORE_ACTDR6_CODE=1               # to run EXAMPLE_EMUL_(EVALUATE/MCMC/NAUTILUS/EMCEE1).yaml
      #export IGNORE_ACTDR6_DATA=1         
      #export IGNORE_NAUTILUS_SAMPLER_CODE=1     # to run PROJECTS/EXAMPLE/EXAMPLE_EMUL_NAUTILUS1.py
      #export IGNORE_POLYCHORD_SAMPLER_CODE=1    # to run PROJECTS/EXAMPLE/EXAMPLE_EMUL_POLY1.yaml
      #export IGNORE_GETDIST_CODE=1              # to run EXAMPLE_TENSION_METRICS.ipynb
      #export IGNORE_TENSIOMETER_CODE=1          # to run EXAMPLE_TENSION_METRICS.ipynb
      
> [!TIP]
> What if users have not configured ML-related keys before sourcing `setup_cocoa.sh`?
> 
> Answer: Comment the keys below before rerunning `setup_cocoa.sh`.
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     # These keys are only relevant if you run setup_cocoa multiple times
>     #export OVERWRITE_EXISTING_ALL_PACKAGES=1    
>     #export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 
>     #export REDOWNLOAD_EXISTING_ALL_DATA=1

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Ensure OpenMP is **OFF**.
    
    export OMP_NUM_THREADS=1
    
 **Step :three:** Run `cobaya-run` on the first emulator example following the commands below.

 - **One model evaluation**:

  - Linux
    
        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EVALUATE1.yaml -f

  - macOS (arm)
 
         mpirun -n 1 --oversubscribe  cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux
    
        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL_MCMC1.yaml -r

- **Halofit Comparison**

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[1-4].py`.

  <p align="center">
  <img width="750" height="750" alt="project_roman_real_plot_halofit_comparison_1" src="https://github.com/user-attachments/assets/e9779555-e27c-4d19-a1f0-0533267762ab" />
  </p>

- **PolyChord**:

  - Linux
    
        mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_POLY1.yaml -r

  - macOS (arm)

        mpirun -n 12 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL_POLY1.yaml -r

- **Nautilus**:

  - Linux
    
        mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python -m mpi4py.futures ./projects/roman_real/EXAMPLE_EMUL_NAUTILUS1.py \
                --root ./projects/roman_real/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
                --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5

  - macOS (arm)

        mpirun -n 12 --oversubscribe python -m mpi4py.futures ./projects/roman_real/EXAMPLE_EMUL_NAUTILUS1.py \
                --root ./projects/roman_real/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
                --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5

- **Emcee**:

  - Linux
    
        mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python ./projects/roman_real/EXAMPLE_EMUL_EMCEE1.py --root ./projects/roman_real/ \
                --outroot "EXAMPLE_EMUL_EMCEE1" --maxfeval 1000000

  - macOS (arm)

        mpirun -n 12 --oversubscribe python ./projects/roman_real/EXAMPLE_EMUL_EMCEE1.py \
            --root ./projects/roman_real/ --outroot "EXAMPLE_EMUL_EMCEE1" --maxfeval 1000000

  The number of steps per MPI worker is $n_{\\rm sw} =  {\\rm maxfeval}/n_{\\rm w}$,
  with the number of walkers being $n_{\\rm w}={\\rm max}(3n_{\\rm params},n_{\\rm MPI})$.
  For proper convergence, each walker should traverse 50 times the autocorrelation length ($\tau$),
  which is provided in the header of the output chain file. A reasonable rule of thumb is to assume
  $\tau > 200$ and therefore set ${\\rm maxfeval} > 10,000 \times n_{\\rm w}$.
  Finally, our code sets burn-in (per walker) at $5 \times \tau$.

  With these numbers, users may ask when `Emcee` is preferable to `Metropolis-Hastings`?
  Here are a few numbers based on our `Planck CMB (l < 396) + SN + BAO + LSST-Y1` test case.
  1) `MH` achieves convergence with $n_{\\rm sw} \sim 150,000$ (number of steps per walker), but only requires four walkers.
  2) `Emcee` has $\tau \sim 300$, so it requires $n_{\\rm sw} \sim 15,000$ when running with $n_{\\rm w}=114$.
  
  Conclusion: `Emcee` requires $\sim 3$ more evaluations in this case, but the number of evaluations per MPI worker (assuming one MPI worker per walker) is reduced by $\sim 10$.
  Therefore, `Emcee` seems well-suited for chains where the evaluation of a single cosmology is time-consuming (and there is no slow/fast decomposition).

  What if the user runs an `Emcee` chain with `maxeval` insufficient for convergence? `Emcee` saves the chain checkpoint at `chains/outroot.h5`.

- **Sampler Comparison**

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[1-4].py`.

  <p align="center">
  <img width="750" height="750" alt="project_roman_real_plot_sampler_comparison_1" src="https://github.com/user-attachments/assets/bad7e26a-fc2b-4370-a11a-1b6aefd0bab9" />
  </p>
  
- **Global Minimizer**:

  Our minimizer is a reimplementation of `Procoli`, developed by Karwal et al (arXiv:2401.14225) 

  - Linux
    
        mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python ./projects/roman_real/EXAMPLE_EMUL_MINIMIZE1.py --root ./projects/roman_real/ \
                --outroot "EXAMPLE_EMUL_MIN1" --nstw 450

  - macOS (arm)

        mpirun -n 12 python ./projects/roman_real/EXAMPLE_EMUL_MINIMIZE1.py --root ./projects/roman_real/ \
              --outroot "EXAMPLE_EMUL_MIN1" --nstw 450

  The number of steps per Emcee walker per temperature is $n_{\\rm stw}$,
  and the number of walkers is $n_{\\rm w}={\\rm max}(3n_{\\rm params},n_{\\rm MPI})$.
  The minimum number of total evaluations is $3n_{\\rm params} \times n_{\rm T} \times n_{\\rm stw}$, which can be distributed among $n_{\\rm MPI} = 3n_{\\rm params}$ MPI processes for faster results.
    
  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_MIN_COMPARE_CONV[1-2].py`

  <p align="center">
  <img width="750" height="500" alt="compare_min_roman_real" src="https://github.com/user-attachments/assets/1db89ed7-c186-4967-85a6-1f9b59e1f3ce" />
  </p>

  In our testing, $n_{\\rm stw} \sim 250$ worked reasonably well up to $n_{\rm param} \sim \mathcal{O}(10)$.

- **Profile**: 

  - Linux
    
          mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python ./projects/roman_real/EXAMPLE_EMUL_PROFILE1.py \
              --root ./projects/roman_real/ --cov 'chains/EXAMPLE_EMUL_MCMC1.covmat' \
              --outroot "EXAMPLE_EMUL_PROFILE1" --factor 3 --nstw 450 --numpts 10 \
              --profile ${SLURM_ARRAY_TASK_ID} \
              --minfile="./projects/roman_real/chains/EXAMPLE_EMUL_MIN1.txt"

  -  macOS (arm)

          mpirun -n 51 --oversubscribe python ./projects/roman_real/EXAMPLE_EMUL_PROFILE1.py \
              --root ./projects/roman_real/ --cov 'chains/EXAMPLE_EMUL_MCMC1.covmat' \
              --outroot "EXAMPLE_EMUL_PROFILE1" --factor 3 --nstw 450 --numpts 10 \
              --profile ${SLURM_ARRAY_TASK_ID} \
              --minfile="./projects/roman_real/chains/EXAMPLE_EMUL_MIN1.txt"

  The argument `factor` specifies the start and end of the parameter being profiled:

      start value ~ mininum value - factor*np.sqrt(np.diag(cov))
      end   value ~ mininum value + factor*np.sqrt(np.diag(cov))

  We advise ${\rm factor} \sim 3$ for parameters that are well constrained by the data when a covariance matrix is provided.
  If `cov` is not supplied, the code estimates one internally from the prior.
  If a parameter is poorly constrained or `cov` is not given, we recommend ${\rm factor} \ll 1$.

  The script of the plot below is provided at `projects/roman_real/scripts/EXAMPLE_PLOT_PROFILE[1-2].py`

  Profile 1: `Cosmic Shear only (plus weak Gaussian priors)`

  <p align="center">
  <img width="750" height="500" alt="example_lssty1_profile1" src="https://github.com/user-attachments/assets/2fea9d3c-524a-49d9-ae89-cb2bb26594e9" />
  </p>

> [!Warning]
> When running Profiles, you should not set flat priors on parameters that are not well constrained by the data. 
> By doing that, you then risk having the minimizer select values near the boundary of parameter space. This is a big problem when using emulators, as volume near the
> boundary will be inevitable outside the training range. You can convert a flat prior to a Gaussian one by setting the standard deviation to be $\sigma^2 = (hi - lo)^2/12$,
> where $(lo, hi)$ are the flat prior boundaries. In our scripts, we implement a truncated Gaussian prior by adding the following prior block
>
>      prior:
>        # These priors are meant to prevent the sampler to wander far off training
>        g1: "lambda As_1e9: stats.norm.logpdf(As_1e9, loc=2.35, scale=1.6)"
>        g2: "lambda ns: stats.norm.logpdf(ns, loc=0.96, scale=0.05)"
>        g3: "lambda H0: stats.norm.logpdf(H0, loc=70, scale=10.0)"
>        g4: "lambda omegab: stats.norm.logpdf(omegab, loc=0.045, scale=0.012)"
>        g5: "lambda omegam: stats.norm.logpdf(omegam, loc=0.3 , scale=0.25)"
>        g6: "lambda w0pwa: stats.norm.logpdf(w0pwa, loc=-1.0 , scale=1.44)"
>        g7: "lambda w: stats.norm.logpdf(w, loc=-1.0, scale=1.44)"
>        g8: "lambda roman_A1_1: stats.norm.logpdf(roman_A1_1, loc=0, scale=2.5)"
>        g9: "lambda roman_A1_2: stats.norm.logpdf(roman_A1_2, loc=-1.7, scale=2.5)" 
>
> Running Profile also requires emulators trained on larger volumes of the parameter space. 

# Running Hybrid Cosmolike-ML emulators <a name="roman_examples_emul2"></a>

> [!Warning]
> The code and examples associated with this section are still in alpha stage

Our main line of research involves emulators that simulate the entire Cosmolike data vectors, and each project (LSST, Roman, DES) contains its own README with emulator examples. The speed of such emulators is incredible, especially when GPUs are available, and our emulators do take advantage of the CPU-GPU integration on Apple MX chips. For example, the average timing of lsst-y1 cosmic shear data vector emulation is around 0.005s ($\sim$ 200828 evaluations in $\sim$ 850.5 seconds) on a macOS M2 Pro.

While the data vector emulators are incredibly fast, there is an intermediate approach that emulates only the Boltzmann outputs (comoving distance and linear and nonlinear matter power spectra). This hybrid-ML approach can offer greater flexibility, especially in the initial phases of a research project, because changes to the modeling of nuisance parameters or to the assumed galaxy distributions do not require retraining the network. 

Examples in the hybrid case all have the prefix **EXAMPLE_EMUL2** (note the `2`). The required flags in `set_installation_options.sh` are similar to those shown in the previous emulator section.

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores. Below, we set it to 4, the ideal setting for hybrid examples.

  - Linux
    
        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=close; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE

  - macOS (arm)
    
        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=disabled; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE
    
 **Step :three:** Run `cobaya-run` on the first emulator example, following the commands below (here we only provide lsst-y1 examples).

- **One model evaluation**:

  - Linux
    
        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_EVALUATE1.yaml -f

  - macOS (arm)
    
        mpirun -n 1 --oversubscribe  cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_EVALUATE1.yaml -f
    
- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux
    
        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
            --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EMUL2_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_MCMC1.yaml -r
    
Details on the matter power spectrum emulator designs will be presented in the [emulator_code](https://github.com/CosmoLike/emulators_code) repository. Basically, we apply standard neural network techniques to generalize the *syren-new* Eq. 6 of [arXiv:2410.14623](https://arxiv.org/abs/2410.14623) formula for the linear power spectrum (w0waCDM with a fixed neutrino mass of $0.06$ eV) to new models, extended ranges, or higher precision. Similarly, we use networks to generalize the *syren-Halofit* LCDM nonlinear boost fit (Eq. 11 of [arXiv:2402.17492](https://arxiv.org/abs/2402.17492)).

> [!NOTE] 
> Users can decide not to correct the *syren-new* formula for the linear power spectrum (flag in the yaml). Although we have not conducted extensive studies of the caveats of the syren-new approximation, it appears sufficient for w0waCDM forecasts when combined with the Euclid Emulator to compute the nonlinear boost.
>
> For back-of-the-envelope LCDM calculations (e.g., to test cosmolike features), users can also choose not to correct the *syren-Halofit* formula for the LCDM nonlinear boost (see figure below). In this case, the overhead on top of cosmolike computations is minimum, at the order of $0.01$ seconds on a macOS M2Pro laptop. 

- **Emulator Comparison**

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[1-4].py`.

  <p align="center">
  <img width="750" height="750" alt="project_roman_real_plot_sampler_comparison_1" src="https://github.com/user-attachments/assets/a85b3fcc-d82c-4c4a-9341-677445a03dd0" />
  </p>

# Training Roman ML emulators <a name="roman_train__emul"></a>

Emulators for Roman are stored on `emulators/`, we usually provide the YAML filed used to train them. For example, datavectors required to train an emulator 
with settings similar to

    nla_cosmic_shear/w0wa_takahashi_cs_CNN.h5
    nla_cosmic_shear/w0wa_takahashi_cs_CNN.h5

can be found at

    nla_cosmic_shear/w0wa_takahashi_cs_CNN.yaml

The repository `emulators_code` provides the script `dataset_generator_lensing.py` that can generate data vectors for any cosmological project. 

## Compute data vectors to train a cosmic shear data vector emulator

   The script below computes data vectors for cosmic shear (NLA, $w_0w_a$ model and Halofit.

       mpirun -n 10 --oversubscribe \
         python external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
           --root projects/roman_real/  \
           --fileroot emulators/nla_cosmic_shear/ \
           --nparams 10000 \
           --yaml 'w0wa_takahashi_cs_CNN.yaml' \
           --datavsfile 'w0wa_takahashi_dvs_train' \
           --paramfile 'w0wa_takahashi_params_train' \
           --failfile  'w0wa_takahashi_params_failed_train' \
           --chain 0 \
           --unif 0 \
           --temp 64 \
           --maxcorr 0.15 \
           --freqchk 2000 \
           --loadchk 0 \
           --append 1 

- The requested number of data vectors is given by the `--nparams` flag.

- There are two possible samplings.
  - The option `--unif 1` sets the sampling to follow a uniform distribution (respecting parameter boundaries set in the YAML file)
  - The option `--unif 0` sets the sampling to follow a Gaussian distribution with the following options
    -  The covariance matrix is set in the YAML file (keyword `params_covmat_file` inside the `train_args` block).
       For example, our provided YAML selects the Fisher-based *w0wa_fisher_covmat.txt* covariance matrix
    -  Temperature reduces the curvature of the likelihood (`cov = cov/T`) and is set by `--temp` flag 
    -  The correlations of the original covariance matrix are reduced to be less than `--maxcorr`.

- For visualization purposes, setting `--chain 1` sets the script to generate the training parameters without computing the data vectors.

- The output files are

      # Distribution of training points ready to be plotted by GetDist
      w0wa_params_train_cs_64.1.txt
      w0wa_params_train_cs_64.covmat
      w0wa_params_train_cs_64.paramnames
      w0wa_params_train_cs_64.ranges

      #Corresponding data vectors
      w0wa_takahashi_nobaryon_dvs_train_cs_64.npy
      # Training parameters in which the data vector computation failed
      w0wa_params_failed_train_cs_64.txt

- The flags `--freqchk`, `--loadchk`, and `--append` are related to checkpoints. 
  - The option `--freqchk` sets the frequency at which the code saves checkpoints (chk).
  - The options `--loadchk` and `--append` specify whether the code loads the parameters and data vectors from a chk.
    In the two cases below, the code determines which remaining data vectors to compute based on the flags saved in the `--failfile` file.
      - Case 1 (`--loadchk 1` and `--append 1`): the code loads params from the chk and appends `~nparams` models to it. 
      - Case 2 (`--loadchk 1` and `--append 0`): the code loads the params.
  
> [!Warning]
> Computing data vectors for training requires so many MPI processes that, assuming here an HPC environment, it is advantageous to 
> distribute the job among multiple compute nodes. Running Cocoa on multiple nodes requires a few modifications to the MPI command, as indicated below:
>
> - Replace `--mca btl vader,tcp,self` with `--mca btl tcp,self` (vader is a fast communication protocol that only works in a single node)
> - Add the line `-x PATH -x LD_LIBRARY_PATH -x CONDA_PREFIX -x ROOTDIR -x OMP_NUM_THREADS \` 
> - replace `python` by full path `${ROOTDIR}/.local/bin/python`
> - replace mpirun by full path "${CONDA_PREFIX}"/bin/mpirun
> 
> The SLURM script [scripts/random_scripts_used_by_dev/EXAMPLE_EMUL_TRAINX_SBU.sbatch](scripts/random_scripts_used_by_dev/EXAMPLE_EMUL_TRAINX_SBU.sbatch]) exemplify such changes

 
