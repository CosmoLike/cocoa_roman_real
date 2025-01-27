#!/bin/bash

##SBATCH --job-name=dv_array
#SBATCH --job-name=emu_train
#SBATCH --output=/groups/behroozi/hbowden/cocoa_gpu/Cocoa/projects/roman/emulator_output/dv_chain_logs/dv-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
##SBATCH --array=0-4

#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_behroozi
#SBATCH --account=behroozi

#SBATCH -t 12:00:00

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS

module purge > /dev/null 2>&1
#module load anaconda
#conda init bash
#source ~/.bashrc
cd /groups/behroozi/hbowden/cocoa_gpu/Cocoa

source activate cocoa
source start_cocoa.sh
echo $CONDA_PREFIX

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

##module load gsl
  
export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

#export OMP_NUM_THREADS=1

echo calculating dvs

mpirun -n 1 \
--mca btl tcp,self \
--bind-to none \
--map-by numa:pe=${OMP_NUM_THREADS} \
python3 projects/roman/train_emu.py \
projects/roman/Calc_dvs_cs_n8.yaml

#python3 projects/roman/calc_dvs.py \
#projects/roman/Calc_dvs_cs_n8.yaml \
#${SLURM_ARRAY_TASK_ID} 4

#python3 projects/roman/just_OmegaM_dvs.py \
#projects/roman/Calc_dvs_cs_n8.yaml