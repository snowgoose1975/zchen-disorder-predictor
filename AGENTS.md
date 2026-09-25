# My predictor project rules

This directory contains the own intrinsic-disorder prediction head and its input adapters.

- Keep it separate from external baseline implementations.
- Preserve sequence IDs, residue order, and residue numbering.
- Label value 2 means unknown and must be excluded from loss and metrics.
- Record seed, embedding family/layer, model configuration, data manifest, and command.
