This folder holds large binary assets that are not tracked in git (see top-level
.gitignore — `posedreamer/weights/` is excluded).

Expected contents (regenerate / re-download as needed):

- SMPLX_NEUTRAL.npz, smplx/, smplx_kid_template.npy
    SMPL-X model + neutral kid template. Download from https://smpl-x.is.tue.mpg.de/
- smpl/, smpl_kid_template.npy
    SMPL model. Download from https://smpl.is.tue.mpg.de/
- model_transfer/smplx2smpl_deftrafo_setup.pkl
    SMPL-X -> SMPL deformation transfer. From the SMPL-X release.
- model_transfer/smpl2smplx_deftrafo_setup.pkl
    SMPL -> SMPL-X deformation transfer (same release).
- mds_d=256.npy
    SMPL vertex embedding for CSE.
    URL: https://dl.fbaipublicfiles.com/densepose/data/cse/mds_d=256.npy
- min_dist.npy
    Index map from the DensePose CSE vertex space (27554) to the nearest
    SMPL vertex (6890); used when rendering DensePose-COCO annotations.

Generated locally (regenerate with the script noted):
- new_colormap_smpl.npy, new_colormap_smplx.npy
    Densepose-style per-vertex colour maps. Regenerate with:
        python posedreamer/rendering/make_colormaps.py
    (see `create_new_colormap` — it computes the mapping and `np.save`s both
     SMPL and SMPL-X colormaps into this folder.)
