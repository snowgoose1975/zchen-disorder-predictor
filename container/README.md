# Container workflow

The image entrypoint is Python module mode. Build locally and keep the image
private until the benchmark is reviewed.

## CPU image

~~~bash
cd /home/czc12/projects/my-predictor
docker build -f Dockerfile.cpu -t my-disorder-predictor:cpu .
docker run --rm my-disorder-predictor:cpu scripts.predict --help
~~~

## GPU image

~~~bash
docker build -f Dockerfile.gpu -t my-disorder-predictor:gpu .
docker run --rm --gpus all my-disorder-predictor:gpu scripts.train --help
~~~

The GPU base image is selected for CUDA 12.4 compatibility with the A100
deployment target. The actual star run must still use a valid Slurm GPU
allocation and Apptainer --nv.

The previous local rebuild is my-disorder-predictor:gpu-esm2-v16 (with a matching CPU image). It predates the latest TCN padding fix. Rebuild the current source as v17 before formal use. Because the
Dockerfile already sets ENTRYPOINT ["python", "-m"], invoke modules as
docker run IMAGE scripts.predict --help, not as docker run IMAGE python -m
scripts.predict --help.

## Export for star

~~~bash
docker save my-disorder-predictor:gpu -o my-disorder-predictor-gpu.tar
apptainer build my-disorder-predictor-gpu.sif docker-archive://my-disorder-predictor-gpu.tar
~~~

Upload the SIF into the user's own container directory. Do not use the star
Docker daemon and do not publish the image without explicit approval.

The v16 source includes the conv_bigru head, multi-PLM fusion manifest builder/loader, optional protein-balanced loss (loss-weighting protein), and protein/residue-level strata evaluation; it does not add new model weights.
