# Container workflow

This directory documents the portable container workflow. It does not contain
dataset paths, host-specific instructions, or deployment credentials.

The image entrypoint is Python module mode. The public Docker Hub images are
the easiest way to run the project; cloning this repository is only needed for
source inspection, local development, or rebuilding the image.

## Pull a published image

Choose the image that matches the available hardware:

~~~bash
docker pull thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1
docker pull thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1
~~~

Check the command-line interface without mounting any data:

~~~bash
docker run --rm \
  thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1 \
  scripts.predict --help
~~~

For a GPU run, use the GPU tag and expose the host GPU:

~~~bash
docker run --rm --gpus all \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  scripts.train --help
~~~

## Build locally

Cloning the repository is required only when building or modifying the image.
Run these commands from the repository root:

~~~bash
docker build -f Dockerfile.cpu -t my-disorder-predictor:cpu .
docker run --rm my-disorder-predictor:cpu scripts.predict --help

docker build -f Dockerfile.gpu -t my-disorder-predictor:gpu .
docker run --rm --gpus all my-disorder-predictor:gpu scripts.train --help
~~~

The CPU and GPU images contain the same predictor source code. They differ in
their PyTorch/CUDA runtime. The GPU image requires a compatible NVIDIA runtime
on the host.

## Mount data and outputs

Input data should be mounted read-only. Manifests, predictions, metrics, and
checkpoints should be mounted to a separate writable output directory:

~~~bash
docker run --rm \
  -v "$PWD/data/handoff:/data/handoff:ro" \
  -v "$PWD/manifests:/work/manifests:ro" \
  -v "$PWD/outputs:/work/outputs" \
  thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1 \
  scripts.predict_manifest \
  --checkpoint /work/outputs/checkpoint/best.pt \
  --manifest /work/manifests/test.tsv \
  --output /work/outputs/test-predictions.tsv \
  --device cpu
~~~

Replace the paths and arguments with the files in your own environment. Do
not place private datasets, model checkpoints, credentials, or generated
outputs in this repository.

## Archive conversion

If a Docker daemon is unavailable on the execution machine, export the image
as an archive and convert it with Apptainer on a machine that supports that
workflow:

~~~bash
docker save thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1 \
  -o predictor-cpu-esm2-v1.tar

apptainer build predictor-cpu-esm2-v1.sif \
  docker-archive://predictor-cpu-esm2-v1.tar
~~~

The resulting SIF is a portable copy of the image. The exact execution
command depends on the target machine's scheduler and filesystem policy, so
those site-specific instructions should be documented separately from this
public repository.
