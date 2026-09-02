# Hard sample mining

Curriculum-based sample selection (paper §3.3): a LightGBM regressor predicts
the OKS an HMR model achieves on a pose directly from its SMPL-X parameters,
so hard poses can be selected for generation without generating images first.

- `pipeline.py` — end-to-end: mines SMPL-X parameters from `.h5` label files,
  trains the model, evaluates on a held-out split.

  ```bash
  python pipeline.py --h5_folder_path /path/to/h5/files --save_folder ./results
  ```

- `data_mining.py` — builds the (SMPL-X params → OKS) training table.
- `train_model.py` — LightGBM model definition and training.
- `evaluator.py` — test-set evaluation and per-sample analysis.
- `inference.py` — rank new candidate poses by predicted difficulty; the
  lowest-predicted-OKS samples form the stage-2 generation set.
