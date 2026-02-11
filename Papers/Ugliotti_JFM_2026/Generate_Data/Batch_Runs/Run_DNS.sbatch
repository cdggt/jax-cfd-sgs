#!/bin/bash
#SBATCH --account=
#SBATCH -J jax_lcr_ensemble
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -q inferno
#SBATCH --array=1-12
#SBATCH -o logs/LCR_Ensemble-%A_%a.out
#SBATCH -e logs/LCR_Ensemble-%A_%a.err

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

# Avoid mixing in ~/.local/pythonX.Y
export PYTHONNOUSERSITE=1

module purge
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || module load miniforge3 2>/dev/null

# Enable conda in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate your prefix env
conda activate /storage/home/hcoda1/4/mugliotti3/r-rg119-0/environments/JAX

echo "Node: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Python: $(which python)"
python -c "import sys; print(sys.executable); print(sys.version)"

# (Optional) sanity checks
python -c "import jax; print('jax', jax.__version__)"
python -c "import jax_cfd; print('jax_cfd OK')"
python -c "import optax; print('optax OK')"

echo "Starting ensemble member ${SLURM_ARRAY_TASK_ID}..."
srun -u python batched_forced_dns.py --node_id ${SLURM_ARRAY_TASK_ID}
echo "Member ${SLURM_ARRAY_TASK_ID} done."
