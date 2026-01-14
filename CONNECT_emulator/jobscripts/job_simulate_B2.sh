#!/bin/bash
#SBATCH --exclusive
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=8
#SBATCH --partition=fat_genoa
#SBATCH --time=120:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
echo "========= Job started at `date` =========="

# activate proper environment if needed
source activate ConnectEnvironment

module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load FlexiBLAS/3.4.4-GCC-13.3.0
export OMPI_MCA_pml=ucx

omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

# source planck data (load path from connect.conf)
clik_line=$(grep -hr "clik" mcmc_plugin/connect.conf)
path_split=(${clik_line//= / })
path="$(echo ${path_split[1]} | sed "s/'//g")/bin/clik_profile.sh"
source $path


python connect.py create input/dncdm_to_ncdm_DR_B2.param

echo "========= Job finished at `date` =========="
