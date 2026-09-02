# Rendering (mesh → RGB control images)

Implements the PNCC-style mesh-to-RGB encoding used to condition the
diffusion model (paper §3.1, "Image Generation"): each spatial axis of the
canonical mesh is normalized independently and mapped to an RGB channel.

- `make_colormaps.py` — run once to compute the per-vertex colormaps and save
  `new_colormap_smpl.npy` / `new_colormap_smplx.npy` into
  [`../weights/`](../weights/README.txt).
- `renderer_pyrd.py` — the pyrender-based renderer used by every render
  script (loads the colormaps from `../weights/`).
- `smplx_dataset.py` — dataset wrapper batching SMPL-X params for rendering.
- `render_smplx_to_cse.py` — renders SMPL-X parameter files into control
  images.

Requires the SMPL-X model files and CSE embedding in `../weights/` — see
[`../weights/README.txt`](../weights/README.txt).
