#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --partition=rome
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

source activate ConnectEnvironment

module load 2024
module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load FlexiBLAS/3.4.4-GCC-13.3.0

export OMPI_MCA_pml=ucx

omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

# source planck data (load path from connect.conf)
clik_line=$(grep -hr "clik" $HOME/connect_public/mcmc_plugin/connect.conf)
path_split=(${clik_line//= / })
path="$(echo ${path_split[1]} | sed "s/'//g")/bin/clik_profile.sh"
source $path

cd $HOME/montepython_public/

mpirun -np 16 python3.7 montepython/MontePython.py run -p input/Pl18TTTEEE_lens_bao_DESIY3_Mnu_dec_toDR_connect.param -o chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_dec_toDR_connect -b chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_dec_toDR/Pl18TTTEEE_lens_bao_DESIY3_Mnu_dec_toDR.bestfit -c chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_dec_toDR/Pl18TTTEEE_lens_bao_DESIY3_Mnu_dec_toDR.covmat -f 1.8 -N 1000000 --conf $HOME/connect_public/mcmc_plugin/connect.conf
#mpirun -np 16 python3.7 montepython/MontePython.py run -p input/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B1_dec_toNCDM_connect.param -o chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B1_dec_toNCDM_connect -b chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B1/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B1_v22.bestfit -c chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B1/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B1_v22.covmat -f 0.5 -N 1000000 --conf $HOME/connect_public/mcmc_plugin/connect.conf
#mpirun -np 16 python3.7 montepython/MontePython.py run -p input/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B2_dec_toNCDM_connect.param -o chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B2_dec_toNCDM_connect -b chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B2/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B2_v22.bestfit -c chains/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B2/Pl18TTTEEE_lens_bao_DESIY3_Mnu_B2_v22.covmat -f 0.5 -N 1000000 --conf $HOME/connect_public/mcmc_plugin/connect.conf
