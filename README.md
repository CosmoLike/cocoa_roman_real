## Running Cosmolike projects <a name="running_cosmolike_projects"></a> 
(From lsst_y1 repository)

In this tutorial, we assume the user installed Cocoa via the *Conda installation* method, and the name of the Conda environment is `cocoa`. We also presume the user's terminal is in the folder where Cocoa was cloned.

 **Step :one:**: activate the cocoa Conda environment, go to the `cocoa/Cocoa/projects` folder, and clone this project:
    
      conda activate cocoa

and

      cd ./cocoa/Cocoa/projects

and

      ${CONDA_PREFIX}/bin/git clone --depth 1 https://github.com/CosmoLike/cocoa_roman_real.git --branch hb_roman roman_real

:warning: Cocoa scripts and YAML files assume the removal of the `cocoa_` prefix when cloning the repository.
      
 **Step :two:**: go back to the Cocoa main folder and activate the private Python environment
    
      cd ../

and

      source start_cocoa.sh
 
:warning: Remember to run the `start_cocoa.sh` shell script only **after cloning** the project repository (or if you already in the `(.local)` environment, run `start_cocoa.sh` again). 

**Step :three:**: compile the project
 
      source ./projects/roman_real/scripts/compile_roman_real.sh

:interrobang: The script `compile_cocoa.sh` also compiles every Cosmolike project on the `cocoa/Cocoa/projects/` folder.

**Step :four:**: select the number of OpenMP cores (below, we set it to 4), and run a template YAML file

    
      export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8
      
One model evaluation:
      
      mpirun -n 1 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/roman_real/EXAMPLE_EVAL1.yaml -f
 
MCMC:

      mpirun -n 4 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/roman_real/EXAMPLE_MCMC1.yaml -f
      

## Deleting Cosmolike projects <a name="running_cosmolike_projects"></a>

Do not delete the `roman_real` folder from the project folder without first running the shell script `stop_cocoa.sh`. Otherwise, Cocoa will have ill-defined soft links. 

:interrobang: Where the ill-defined soft links will be located? 
     
     Cocoa/cobaya/cobaya/likelihoods/
     Cocoa/external_modules/code/
     Cocoa/external_modules/data/ 
    
:interrobang: Why does Cocoa behave like this? The shell script `start_cocoa.sh` creates symbolic links so Cobaya can see the likelihood and data files. Cocoa also adds the Cobaya-Cosmolike interface of all cosmolike-related projects to the `LD_LIBRARY_PATH` and `PYTHONPATH` environmental variables.
