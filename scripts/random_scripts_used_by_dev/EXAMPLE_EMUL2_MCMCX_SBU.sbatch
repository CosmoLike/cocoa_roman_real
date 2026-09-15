#!/bin/bash

#SBATCH --job-name=MC
#SBATCH --output=MC-%A-%a.out
#SBATCH --error=MC-%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=4   # trying to reserve the socket that has 48 cores
#SBATCH --cpus-per-task=12      # trying to reserve the socket that has 48 cores
#SBATCH --time=72:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
if [ -z "${ROOTDIR}" ]; then
  sleep $(( 5 + SLURM_ARRAY_TASK_ID*25 )) # help avoid different scripts running start_cocoa.sh simultaneously
                                          # clash can still happen when one script runs start_cocoa.sh, the other
                                          # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                                          # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 
export MKL_NUM_THREADS=1

"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe \
  --mca pml ob1 --mca btl vader,tcp,self \
  --bind-to core:overload-allowed --report-bindings \
  --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
  cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_MCMC${SLURM_ARRAY_TASK_ID}.yaml -r
