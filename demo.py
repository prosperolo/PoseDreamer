"""PoseDreamer demo.

Renders a mesh-to-RGB control image from an SMPL-X pose and (optionally)
generates a photorealistic image from it with the FLUX + EasyControl model.

Render only (needs the assets in posedreamer/weights/, see its README.txt):

    python demo.py

Render + generate (needs a GPU, the EasyControl clone at the repo root, and
access to black-forest-labs/FLUX.1-dev on HuggingFace, or a local copy passed
with --flux_path):

    python demo.py --generate --prompt "a climber on a sunlit granite wall"

The SMPL-X pose can be swapped with --pose /path/to/pose.npz (keys:
body_pose (63,), optionally global_orient (3,), betas (10,), transl (3,)).
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

from posedreamer.utils.paths import REPO_ROOT, WEIGHTS_DIR

# Release location of the trained control LoRA
CONTROL_LORA_REPO = "prosperolo/PoseDreamer"
CONTROL_LORA_FILE = "control_lora.safetensors"
FLUX_REPO = "black-forest-labs/FLUX.1-dev"

IMG_SIZE = 1024
FOCAL_LENGTH = 1000.0


def demo_pose() -> dict:
    """A simple A-pose with a raised left forearm (a wave)."""
    body_pose = np.zeros((21, 3), dtype=np.float32)
    body_pose[15, 2] = -0.9   # left shoulder: rotate arm down
    body_pose[16, 2] = 0.9    # right shoulder: rotate arm down
    body_pose[17, 2] = -1.2   # left elbow: bend forearm up
    return {
        "body_pose": body_pose.reshape(1, -1),
        # The renderer flips the scene 180° about x (camera convention), so an
        # upright body needs the matching flip in its global orientation.
        "global_orient": np.array([[np.pi, 0.0, 0.0]], dtype=np.float32),
        "betas": np.zeros((1, 10), dtype=np.float32),
        "transl": np.array([[0.0, 0.4, 2.5]], dtype=np.float32),
    }


def load_pose(path: str) -> dict:
    data = np.load(path)
    pose = demo_pose()
    for key in ("body_pose", "global_orient", "betas", "transl", "left_hand_pose",
                "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose", "expression"):
        if key in data:
            pose[key] = np.asarray(data[key], dtype=np.float32).reshape(1, -1)
    # Optional native camera of the fitted sample; without it the synthetic
    # demo camera is used (fitted translations only make sense under the
    # camera they were fitted with).
    for key in ("focal", "princpt", "img_shape"):
        if key in data:
            pose[key] = np.asarray(data[key]).flatten()
    return pose


def square_crop_nonblack(image: Image.Image, margin: float = 0.1) -> Image.Image:
    """Square crop around the non-black region, mirroring the pipeline crops."""
    arr = np.array(image)
    mask = (arr > 0).any(axis=2)
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + margin))
    half = side // 2
    h, w = mask.shape
    y0, x0 = np.clip(cy - half, 0, h - side), np.clip(cx - half, 0, w - side)
    return image.crop((x0, y0, x0 + side, y0 + side))


def render_control_image(pose: dict) -> Image.Image:
    import smplx

    from posedreamer.rendering.renderer_pyrd import Renderer

    model = smplx.create(
        str(WEIGHTS_DIR / "SMPLX_NEUTRAL.npz"),
        model_type="smplx",
        gender="neutral", use_face_contour=False,
        num_betas=10, flat_hand_mean=False,
        num_expression_coeffs=10,
        ext="npz", use_pca=False,
    )
    extras = {k: torch.from_numpy(pose[k]) for k in
              ("left_hand_pose", "right_hand_pose", "jaw_pose", "leye_pose",
               "reye_pose", "expression") if k in pose}
    output = model(
        body_pose=torch.from_numpy(pose["body_pose"]),
        global_orient=torch.from_numpy(pose["global_orient"]),
        betas=torch.from_numpy(pose["betas"]),
        **extras,
    )
    verts = output.vertices.detach() + torch.from_numpy(pose["transl"])[:, None]

    if "focal" in pose:
        focal = float(pose["focal"][0])
        princpt = pose["princpt"]
        img_h, img_w = [int(x) for x in pose["img_shape"]]
    else:
        focal, princpt = FOCAL_LENGTH, None
        img_h = img_w = IMG_SIZE
    renderer = Renderer(focal_length=focal, principal_point=princpt,
                        img_w=img_w, img_h=img_h,
                        faces=model.faces, colormap="smplx")
    render = renderer.render_front_view(verts.cpu().numpy())
    renderer.delete()

    # renderer returns RGB; keep it as-is for PIL (pipeline scripts flip to BGR
    # only because they save through cv2.imwrite)
    control = square_crop_nonblack(Image.fromarray(render))
    return control.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


def generate(control_image: Image.Image, prompt: str, control_lora: str, seed: int,
             dpo_checkpoint: str = None, flux_path: str = FLUX_REPO) -> Image.Image:
    sys.path.insert(0, str(REPO_ROOT / "EasyControl"))
    try:
        from src.pipeline import FluxPipeline
        from src.transformer_flux import FluxTransformer2DModel
        from src.lora_helper import set_single_lora
    except ImportError as e:
        raise SystemExit(
            "Could not import EasyControl — clone it at the repo root first "
            "(see IMPORTED_REPOS.md).") from e

    device = "cuda"
    base_path = flux_path
    pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        base_path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device)
    pipe.to(device)

    set_single_lora(pipe.transformer, control_lora, lora_weights=[1], cond_size=512)

    if dpo_checkpoint is not None:
        # Stack the DPO alignment LoRA on top of the control LoRA
        # (same loading procedure as posedreamer/generation/infer_w_dpo.py)
        from peft import get_peft_model, LoraConfig
        peft_config = LoraConfig(r=128, lora_alpha=128.0,
                                 target_modules=["to_q", "to_k", "to_v", "to_out.0"])
        peft_model = get_peft_model(pipe.transformer, peft_config)
        checkpoint = torch.load(dpo_checkpoint, map_location="cpu", weights_only=False)
        transformed = {
            (f"base_model.model.{k}" if not k.startswith("base_model.") else k).replace(".weight", ".default.weight") if "lora_" in k else k: v
            for k, v in checkpoint.items()
        }
        missing, unexpected = peft_model.load_state_dict(transformed, strict=False)
        n_loaded = sum(1 for k in transformed if "lora_" in k and k not in unexpected)
        if n_loaded == 0:
            raise SystemExit(f"No LoRA tensors from {dpo_checkpoint} matched the transformer "
                             f"({len(unexpected)} unexpected keys) — wrong checkpoint format?")
        pipe.transformer = peft_model
        pipe.to(device)
        print(f"Applied DPO checkpoint: {dpo_checkpoint} "
              f"({n_loaded} LoRA tensors loaded, {len(unexpected)} unexpected keys)")

    result = pipe(
        prompt,
        height=IMG_SIZE,
        width=IMG_SIZE,
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
        spatial_images=[control_image],
        subject_images=[],
        cond_size=512,
    ).images[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose", default="demo_pose.npz",
                        help=".npz with SMPL-X parameters and optional native camera; a LAION example\n"
                             "ships with the repo. Missing file falls back to a built-in synthetic pose.")
    parser.add_argument("--out_dir", default="demo_out")
    parser.add_argument("--generate", action="store_true",
                        help="Also generate an image with FLUX + the control LoRA")
    parser.add_argument("--prompt", default="A soccer player in a red and blue jersey running towards the camera, red athletic socks and cleats, blurred crowd of spectators in the background. DSLR photo, maximum detail")
    parser.add_argument("--control_lora", default=None,
                        help="Path to the control LoRA .safetensors (downloaded from HF if omitted)")
    parser.add_argument("--dpo_checkpoint", default=None,
                        help="Optional DPO alignment checkpoint (.ckpt) to stack on the control LoRA")
    parser.add_argument("--flux_path", default=FLUX_REPO,
                        help="FLUX.1-dev weights: a HuggingFace repo id or a local diffusers-format folder")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pose = load_pose(args.pose) if args.pose and os.path.exists(args.pose) else demo_pose()

    control_image = render_control_image(pose)
    control_path = os.path.join(args.out_dir, "control.png")
    control_image.save(control_path)
    print(f"Control image written to {control_path}")

    if not args.generate:
        return

    control_lora = args.control_lora
    if control_lora is None:
        from huggingface_hub import hf_hub_download
        control_lora = hf_hub_download(repo_id=CONTROL_LORA_REPO, filename=CONTROL_LORA_FILE)

    image = generate(control_image, args.prompt, control_lora, args.seed, args.dpo_checkpoint,
                     flux_path=args.flux_path)
    generated_path = os.path.join(args.out_dir, "generated.png")
    image.save(generated_path)
    print(f"Generated image written to {generated_path}")


if __name__ == "__main__":
    main()
