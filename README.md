# PoseDreamer

**Scalable and Photorealistic Human Data Generation Pipeline with Diffusion Models**

Lorenza Prospero, Orest Kupyn, Ostap Viniavskyi, João F. Henriques, Christian Rupprecht

[**Project page and dataset**](https://prosperolo.github.io/posedreamer)

PoseDreamer generates large-scale synthetic datasets for human mesh recovery
with diffusion models: SMPL-X parameters are sampled first and rendered as
mesh-to-RGB control images, a DPO-aligned FLUX control model generates
photorealistic images from them, and hard-sample mining plus multi-stage
filtering keep the 3D labels and images consistent at scale.

## Repository structure

The package follows the pipeline stages of the paper; each folder has its own
README with usage details.

| Folder | Paper | Stage |
|---|---|---|
| [`posedreamer/label_generation/`](posedreamer/label_generation/README.md) | §3.1 | SMPL-X parameters + captions from LAION and AMASS |
| [`posedreamer/rendering/`](posedreamer/rendering/README.md) | §3.1 | PNCC-style mesh-to-RGB control rendering |
| [`posedreamer/control_dataset/`](posedreamer/control_dataset/README.md) | §3.1 | Control-training pairs from DensePose-COCO and AGORA |
| [`posedreamer/generation/`](posedreamer/generation/README.md) | §3.1–3.2 | FLUX + EasyControl generation and DPO alignment |
| [`posedreamer/hard_mining/`](posedreamer/hard_mining/README.md) | §3.3 | Difficulty-aware sample selection (LightGBM OKS predictor) |
| [`posedreamer/filtering/`](posedreamer/filtering/README.md) | §3.4 | Crowding / OKS / head-pose quality filtering |
| [`posedreamer/metrics/`](posedreamer/metrics/README.md) | §4.4 | IS/FID and distribution-gap evaluation |
| [`posedreamer/smplx_convert/`](posedreamer/smplx_convert/README.md) | — | SMPL-X ↔ SMPL parameter conversion |
| [`posedreamer/weights/`](posedreamer/weights/README.txt) | — | Body models, colormaps, and other assets (not tracked) |

## Installation

```bash
git clone https://github.com/prosperolo/PoseDreamer.git && cd PoseDreamer
pip install -r requirements.txt
pip install -e .
```

Then:

1. Clone the third-party repos used by some stages (`smplx`, `EasyControl`,
   optionally `agora_evaluation`) into the paths described in
   [`IMPORTED_REPOS.md`](IMPORTED_REPOS.md).
2. Download the body models and assets listed in
   [`posedreamer/weights/README.txt`](posedreamer/weights/README.txt), and run
   `python posedreamer/rendering/make_colormaps.py` once to generate the
   mesh-to-RGB colormaps.
3. SMPL-X parameters for LAION images are fitted with the official
   [SMPLer-X](https://github.com/caizhongang/SMPLer-X) and
   [TokenHMR](https://github.com/saidwivedi/TokenHMR) repositories, which are
   not included here.

The SLURM `.sh` files throughout the repo are example launchers: set the
placeholder paths at the top of each script and adapt the `#SBATCH` headers to
your cluster.

## Quick demo

Once `weights/` is set up, render the mesh-to-RGB control image for the
example pose shipped with the repo (`demo_pose.npz`, a running soccer player
fitted from a LAION photo):

```bash
python demo.py
```

With a GPU, the `EasyControl/` clone in place, and access to
`black-forest-labs/FLUX.1-dev`, also generate a photorealistic image from it
(the released control LoRA is downloaded automatically):

```bash
python demo.py --generate
```

Outputs land in `demo_out/`. Useful flags: `--prompt` for your own caption,
`--pose your_pose.npz` for your own SMPL-X parameters (optionally with the
native `focal`/`princpt`/`img_shape` of the fit), `--flux_path` to point at a
local FLUX.1-dev checkout, `--control_lora` for a local control-LoRA
checkpoint, and `--dpo_checkpoint` to stack a DPO alignment LoRA.

## Pipeline overview

1. **Labels** — sample SMPL-X parameters and captions from LAION (scene
   diversity) and AMASS (pose diversity): `label_generation/`.
2. **Control model** — build DensePose-COCO + AGORA control pairs
   (`control_dataset/`), then train the spatial control LoRA
   (`generation/train_spatial.sh`).
3. **DPO alignment** — generate candidates, rank them by OKS, and align the
   control model with Flow-DPO: `generation/alignment/`.
4. **Hard mining** — train the OKS predictor and select challenging poses for
   the final generation round: `hard_mining/`.
5. **Generation & filtering** — generate at scale (`generation/infer_w_dpo.py`)
   and keep only samples passing crowding, OKS, and head-pose checks:
   `filtering/`.

## Citation

```bibtex
@inproceedings{prospero2026posedreamer,
  title     = {PoseDreamer: Scalable and Photorealistic Human Data Generation Pipeline with Diffusion Models},
  author    = {Prospero, Lorenza and Kupyn, Orest and Viniavskyi, Ostap and Henriques, Jo{\~a}o F. and Rupprecht, Christian},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE). Third-party repositories, datasets,
and body models keep their own licenses.
