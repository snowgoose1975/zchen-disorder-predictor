#!/usr/bin/env bash
#SBATCH --job-name=own-disorder-matrix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Submit only after the target cluster has provided a valid account and GPU
# partition. The account and partition names are site-specific.
# sbatch --account=<ACCOUNT> scripts/slurm/run_matrix_gpu.sbatch
set -euo pipefail

: "${PREDICTOR_SIF:?set PREDICTOR_SIF to a user-owned Apptainer SIF}"
WORK_ROOT="${WORK_ROOT:-${PWD}/work}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${WORK_ROOT}/manifests}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/outputs/matrix}"
EMBEDDING_FAMILIES="${EMBEDDING_FAMILIES:-esm2-layer30,prott5-layer24,esmc-layer30}"
ARCHITECTURES="${ARCHITECTURES:-linear,mlp,tcn,bigru,conv_bigru}"
SEEDS="${SEEDS:-20260925,20260926,20260927}"
NORMALIZATION="${NORMALIZATION:-none}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-residue}"

mkdir -p "${OUTPUT_ROOT}"

apptainer exec --nv --cleanenv --pwd /opt/my-predictor \
  --bind "${MANIFEST_ROOT}:/workspace/manifests:ro" \
  --bind "${OUTPUT_ROOT}:/workspace/outputs:rw" \
  "${PREDICTOR_SIF}" \
  python -m scripts.run_matrix \
  --manifest-root /workspace/manifests \
  --output-root /workspace/outputs \
  --embedding-families "${EMBEDDING_FAMILIES}" \
  --architectures "${ARCHITECTURES}" \
  --seeds "${SEEDS}" \
  --embedding-normalization "${NORMALIZATION}" \
  --loss-weighting "${LOSS_WEIGHTING}" \
  --device cuda
