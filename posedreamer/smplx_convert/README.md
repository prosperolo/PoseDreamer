# SMPL-X → SMPL conversion

`convert.py` reads AMASS-format SMPL-X `.npz` files and fits SMPL parameters
(`root_orient`, `body_pose`, `betas`) by combining a SMPL-X forward pass with
the SMPL fitting loop in `transfer_model`.

It is glue around the upstream `transfer_model` package from
[vchoutas/smplx](https://github.com/vchoutas/smplx) — i.e. it reproduces the
behaviour of `transfer_model/write_obj.py` → `transfer_model/__main__.py` →
`transfer_model/merge_output.py` without writing intermediate mesh files.

## Dependency

The `transfer_model` package must be importable. Clone the smplx repo (see
[`../../IMPORTED_REPOS.md`](../../IMPORTED_REPOS.md)) and ensure it is on
`PYTHONPATH`, e.g.:

```bash
export PYTHONPATH=/path/to/repo/smplx:$PYTHONPATH
```

or run from inside the `smplx/` clone.

## Usage

```bash
./convert.sh 0,1,2,3   # GPUs to use
```

`convert.sh` is a launcher that runs `convert.py` in parallel across the
provided GPUs.
