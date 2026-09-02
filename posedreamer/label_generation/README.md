# Label generation

Produces the SMPL-X parameters, control renders, and captions that condition
image generation (paper §3.1, "Label Generation"). Two complementary pose
sources are used:

## LAION (scene diversity)

1. `process_laion.py` (launcher: `process_laion.sh`) — YOLO person detection
   on LAION images, square crops, DensePose predictions
   (`predict_densepose_crops.py` + the two detectron2 configs), and per-crop
   metadata. Outputs `image_crops/`, `densepose/`, `metadata/`.
2. `generate_captions.py` (launcher: `generate_captions.sh`) — VLM captions
   for the crops (`image_captioning.py` wraps BLIP/Gemma-style captioners).
3. SMPL-X parameters for the crops are fitted with the official
   [SMPLer-X](https://github.com/caizhongang/SMPLer-X) and
   [TokenHMR](https://github.com/saidwivedi/TokenHMR) repos (not included
   here); both are run and combined to reduce single-model bias.

## AMASS (pose diversity)

1. `render_amass.py` (launcher: `render_amass.sh`) — samples frames from
   AMASS sequences and renders SMPL-X control images + `.h5` labels using
   `posedreamer.rendering`.
2. `densepose_captions.py` (launcher: `densepose_captions.sh`) — VLM captions
   describing the rendered pose plus a generated scene context.

`caption_processor.py` post-processes captions (it is also used by the
generation stage). `compute_oks_metric.py` scores generated samples against
their labels with OKS (used for DPO pair construction and threshold tuning).
