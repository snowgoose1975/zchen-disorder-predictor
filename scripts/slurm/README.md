# Slurm templates

These files are generic templates only. They are not submitted automatically,
and every target cluster may require different partition, account, time, and
filesystem settings.

Set paths for your own machine before submission:

~~~bash
PREDICTOR_SIF=/path/to/predictor.sif \
WORK_ROOT=/path/to/project/work \
sbatch --account=<ACCOUNT> scripts/slurm/run_matrix_gpu.sbatch
~~~

The template binds manifests read-only and writes only under the configured
output directory. It does not assume a particular host, shared filesystem, or
account name.

Set `EMBEDDING_FAMILIES=fusion-esm2-prott5-esmc` to run only the explicit
multi-PLM fusion matrix, or leave it unset for the three single-family tracks.
Set `LOSS_WEIGHTING=protein` for the protein-balanced training ablation; the
default is residue.
