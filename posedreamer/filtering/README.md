# Dataset filtering

Multi-stage quality filtering of generated samples (paper §3.4). A Hydra
pipeline streams (image, control render, label) triplets through filters and
applies an action (copy/move/no-op) to samples that pass all of them.

```bash
python filter.py action=copy action.dest_base_dir=/path/to/filtered-output
```

Configure inputs in `config/filter.yaml` — each entry maps an image folder to
its control-render and `.h5` label folders.

Filters (`filters/`):
- `keypoint_filter.py` — crowded-scene rejection (YOLO detection count) and
  OKS pose-misalignment rejection against the reprojected SMPL-X joints
  (`functional/oks.py`).
- `head_pose_filter.py` — 3D head-pose consistency: compares roll/pitch/yaw
  from the SMPL-X label against a VGGHeads prediction on the generated image
  (`functional/head_pose.py`). Requires SMPL-X labels.

Actions (`actions.py`, selected via `config/action/`): `copy`, `move`, or
`noop` (statistics only). `filter.sh` is an example SLURM array launcher;
array tasks shard the sample list.
