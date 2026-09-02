# Data directories

- `raw/`: immutable downloads, separated by provider and product.
- `interim/`: decoded, aligned, regridded, and quality-controlled working data.
- `processed/`: labels, catalogue, model features, tensors, and train/validation/test splits.
- `metadata/`: provenance, checksums, variable definitions, and processing history.

Do not manually modify downloaded source files. Every forecast value must retain issue time, valid time, lead, member, provider, and product identifiers.
