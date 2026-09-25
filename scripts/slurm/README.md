# Slurm templates

These files are templates only. They are not submitted automatically.

The GPU matrix job requires a valid Slurm account and a user-owned SIF:

~~~bash
PREDICTOR_SIF=/home/zchen/disorder-baselines/containers/own-predictor-gpu-esm2-v17.sif \
  sbatch --account=<ACCOUNT> scripts/slurm/run_matrix_gpu.sbatch
~~~

Set `EMBEDDING_FAMILIES=fusion-esm2-prott5-esmc` to run only the explicit
multi-PLM fusion matrix, or leave it unset for the three single-family tracks.
The template binds manifests read-only and writes only under the user-owned
output directory. It does not modify the shared handoff.

Set LOSS_WEIGHTING=protein for the protein-balanced training ablation; the default is residue. The v17 evaluator accepts either protein-level or complete residue-level strata files.
