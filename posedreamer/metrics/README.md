# Metrics

Image-quality and distribution-gap evaluation of generated vs. rendered
datasets (paper §4.4).

- `image_quality.py` — Inception Score / FID over a folder of images.
- `distribution_gap.py` — DINO/CLIP feature extraction and t-SNE / UMAP /
  precision-recall-distribution comparisons across datasets (edit the
  `DATASETS` dict at the top to point at your local copies).
- `features_extractors.py`, `func.py`, `graphs.py` — feature extraction, PRD
  computation, and plotting helpers.
