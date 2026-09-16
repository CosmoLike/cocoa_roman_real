#!/bin/bash

#SBATCH --job-name=TREMUL
#SBATCH --output=TREMUL-%A_%a.out
#SBATCH --error TREMUL-%A_%a.out.err
#SBATCH --nodes=6
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --partition=long-40core
#SBATCH --time=48:00:00

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

declare -i AID=${SLURM_ARRAY_TASK_ID:-0}
cd "$SLURM_SUBMIT_DIR"
if [ -z "${ROOTDIR}" ]; then
  sleep $(( 5 + AID*25 )) # help avoid different scripts running start_cocoa.sh simultaneously
                          # clash can still happen when one script runs start_cocoa.sh, the other
                          # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                          # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
else
  cd "${ROOTDIR}" # just to be sure
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

# ---------------------------------------------------------------------------
# index j = int(SLURM_ARRAY_TASK_ID / 3)
# ---------------------------------------------------------------------------
declare -i NY=3
declare -a temp=( "256" "128" "128" )
declare -a mode=( 'train' 'val' 'test' )
declare -a nparams=( '1500000' '50000' '50000' )
declare -a boundary=( '1.0' '0.975' '0.975' )

# ---------------------------------------------------------------------------
# index j = SLURM_ARRAY_TASK_ID % 3
# ---------------------------------------------------------------------------
declare -a datavsfile=( 'w0wa_takahashi_cs_dvs_'
                        'w0wa_hmcode2020_cs_dvs_' )
declare -a paramfile=( 'w0wa_params_takahashi_cs_'
                       'w0wa_params_hmcode2020_cs_' )
declare -a failfile=( 'w0wa_params_failed_takahashi_cs_'
                      'w0wa_params_failed_hmcode2020_cs_' )
declare -a yaml=( 'w0wa_takahashi_cs_cnn.yaml'
                  'w0wa_hmcode2020_cs_cnn.yaml' )
declare -a fileroot=( 'emulators/nla_cosmic_shear'
                      'emulators/nla_cosmic_shear' )

declare -i i=$(( AID / NY ))
declare -i j=$(( AID % NY ))
(( i < ${#datavsfile[@]} )) || exit 0 # padding: skip unused slots

export rootprog="external_modules/code/emulators/emultrf/emultraining"

"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR -x OMP_NUM_THREADS \
  -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
  --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
  --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
  --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
  ${ROOTDIR}/.local/bin/python ${rootprog}/dataset_generator_lensing.py \
    --root projects/roman_real/  \
    --fileroot "${fileroot[$i]}" \
    --nparams "${nparams[$j]}" \
    --yaml "${yaml[$i]}" \
    --datavsfile "${datavsfile[$i]}""${mode[$j]}" \
    --paramfile "${paramfile[$i]}""${mode[$j]}" \
    --failfile "${failfile[$i]}""${mode[$j]}" \
    --chain 0 \
    --unif 0 \
    --temp "${temp[$j]}" \
    --maxcorr 0.15 \
    --freqchk 10000 \
    --loadchk 1 \
    --append 0 \
    --boundary "${boundary[$j]}"
