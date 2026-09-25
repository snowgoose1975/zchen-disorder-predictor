# zchen-disorder-predictor

Residue-level intrinsic-disorder prediction heads for protein sequences and precomputed protein-language-model embeddings.

The project compares small prediction heads while keeping the upstream representation fixed. ESM2 is an optional upstream encoder, not a second disorder predictor.

## Release mapping

The first Docker Hub release is:

```text
thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1
```

It was built from the local development image `my-disorder-predictor:gpu-esm2-v17`. The current Dockerfile does not bundle ESM2 model weights. Precomputed embeddings do not require ESM2 weights, while FASTA-to-embedding inference needs.

The corresponding CPU release can use `thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1`.

## Input contract

The normal training path is:

```text
embedding matrix [L, D] -> prediction head -> residue disorder scores [L]
```

Supported inputs:

1. `.npy` or `.npz` residue embeddings, one vector per residue.
2. FASTA for inference, paired with existing embeddings or an ESM2 model.
3. Tab-separated manifests for training, prediction, and evaluation.

The current handoff uses ESM2 `layer_30` with width 1280. The embedding family and layer must match the checkpoint. Label value `2` means unknown; unknown residues and padding are excluded through the mask.

Training manifests contain at least `id`, `split`, `embedding`, `embedding_key`, `labels`, `mask`, and `length`.

## Prediction heads

```text
linear       residue-wise linear head
mlp          multilayer perceptron head
tcn          temporal convolutional head
bigru        bidirectional GRU sequence-context head
conv_bigru   convolution plus bidirectional GRU head
```

## Docker quick start

```bash
docker pull thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1

docker run --rm --gpus all \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  scripts.predict --help
```

The image entrypoint is `python -m`, so invoke modules as `scripts.predict`, `scripts.train`, and so on. Check CUDA with:

```bash
docker run --rm --gpus all --entrypoint python \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  -c "import torch; print('cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For CPU execution, omit `--gpus all`, use the CPU image, and set `--device cpu`.

## Build a manifest

Mount source data read-only and generated files read-write. Keep the same data mount in later commands because manifest paths are resolved inside the container.

```bash
mkdir -p manifests outputs

docker run --rm --gpus all \
  -v "$PWD/data/handoff:/data/handoff:ro" \
  -v "$PWD/manifests:/work/manifests" \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  scripts.build_manifest \
  --data-root /data/handoff \
  --output-dir /work/manifests/esm2-layer30 \
  --asset-type esm2 \
  --embedding-key layer_30
```

Useful options include `--label-channel merged`, `--label-channel curated`, `--splits train,val,test`, and `--limit N` for a smoke manifest.

## Train

Required arguments are `--train-manifest`, `--val-manifest`, and `--output-dir`. The test split is not used for model selection.

```bash
docker run --rm --gpus all \
  -v "$PWD/data/handoff:/data/handoff:ro" \
  -v "$PWD/manifests:/work/manifests:ro" \
  -v "$PWD/outputs:/work/outputs" \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  scripts.train \
  --train-manifest /work/manifests/esm2-layer30/train.tsv \
  --val-manifest /work/manifests/esm2-layer30/val.tsv \
  --output-dir /work/outputs/mlp-esm2-layer30 \
  --architecture mlp \
  --epochs 10 \
  --batch-size 4 \
  --device cuda \
  --seed 20260925 \
  --num-workers 2 \
  --embedding-normalization none
```

Common optional arguments are `--architecture {linear,mlp,tcn,bigru,conv_bigru}`, `--epochs`, `--batch-size`, `--hidden-dim`, `--dropout`, `--learning-rate`, `--loss-weighting {residue,protein}`, `--embedding-normalization {none,standardize}`, `--seed`, and `--device {auto,cpu,cuda}`.

## Predict a split

Required arguments are `--checkpoint`, `--manifest`, and `--output`.

```bash
docker run --rm --gpus all \
  -v "$PWD/data/handoff:/data/handoff:ro" \
  -v "$PWD/manifests:/work/manifests:ro" \
  -v "$PWD/outputs:/work/outputs" \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  scripts.predict_manifest \
  --checkpoint /work/outputs/mlp-esm2-layer30/best.pt \
  --manifest /work/manifests/esm2-layer30/test.tsv \
  --output /work/outputs/mlp-esm2-layer30/test-predictions.tsv \
  --device cuda
```

## Evaluate

```bash
docker run --rm \
  -v "$PWD/manifests:/work/manifests:ro" \
  -v "$PWD/outputs:/work/outputs" \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  scripts.evaluate \
  --prediction /work/outputs/mlp-esm2-layer30/test-predictions.tsv \
  --manifest /work/manifests/esm2-layer30/test.tsv \
  --output-json /work/outputs/mlp-esm2-layer30/test-metrics.json \
  --output-protein-tsv /work/outputs/mlp-esm2-layer30/test-protein.tsv
```

The evaluator reports masked residue-level AUROC, average precision, Brier score, calibration error, MCC, F1, coverage, positive fraction, and protein-level summaries.

## Single-protein inference

Embedding input skips the ESM2 encoder:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/outputs:/work/outputs" \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  scripts.predict \
  --embedding /work/data/sample.npz \
  --embedding-key layer_30 \
  --checkpoint /work/outputs/mlp-esm2-layer30/best.pt \
  --output /work/outputs/sample-scores.tsv \
  --device cuda
```

FASTA input can use existing per-protein embeddings with `--fasta`, `--embedding-dir`, and `--embedding-key`. For FASTA-to-ESM2 inference, `--esm2-model` must refer to a local model directory or an accessible model identifier, and its layer/width must match the checkpoint.

## Outputs

The main output files are `best.pt` (checkpoint), `config.json` (configuration), `*-predictions.tsv` (continuous residue scores), `*-metrics.json` (aggregate metrics), and `*-protein.tsv` (per-protein metrics). Keep continuous scores for evaluation; do not replace them with thresholded labels.

## Local builds

```bash
docker build -f Dockerfile.cpu -t my-disorder-predictor:cpu-esm2-v17 .
docker build -f Dockerfile.gpu -t my-disorder-predictor:gpu-esm2-v17-offline .
```

## Repository layout

```text
my_predictor/       model, data, embedding, normalization, and metrics code
scripts/            manifest, train, predict, evaluate, and pipeline CLIs
tests/              unit tests
Dockerfile.cpu      CPU runtime image
Dockerfile.gpu      CUDA/PyTorch runtime image
requirements*.txt   Python runtime dependencies
container/          container notes
```

Datasets, checkpoints, generated outputs, Docker archives, virtual environments, and caches are intentionally excluded from Git.
