#!/bin/bash

#SBATCH --account=behroozi
#SBATCH --job-name=dv_chain
#SBATCH --output=emu-%A_%a.out
#SBATCH -e emu-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=5
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_behroozi
#SBATCH -t 12:00:00

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of gpus per task is $SLURM_GPUS_PER_TASK

module purge > /dev/null 2>&1
cd /groups/behroozi/hbowden/cocoa/Cocoa

source activate cocoa
source start_cocoa.sh


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

CONFIGFILE=./projects/roman_real/Calc_dvs_cs_n8.yaml
TRAINEMU=./projects/roman_real/train_emulator.py
#SAMPLEEMU=./projects/desy1xplanck/emulator_deprecated/sample_emulator.py
export OMP_NUM_THREADS=1
$PYTHON3 -u ${TRAINEMU} $CONFIGFILE --overwrite

#python3 projects/roman_real/train_emulator.py \
#./projects/roman_real/test_roman.yaml \
#-f total_samples_lhs_10.npy \
#-o test_emu
