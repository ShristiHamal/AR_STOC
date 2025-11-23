import os
from typing import Optional

import torch
from PIL import Image
from PIL import ImageFilter, ImageDraw

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
)
from controlnet_aux import OpenposeDetector


# -----------------------------
# Environment tweaks
# -----------------------------
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["XFORMERS_DISABLED"] = os.environ.get("XFORMERS_DISABLED", "1")


# -----------------------------
# Auto cloth mask (same idea as before)
# -----------------------------
def generate_cloth_mask(cloth_image: Image.Image) -> Image.Image:
    """
    Auto-generate cloth mask from the clothing image.

    Assumes white / light background and darker clothing.
    White background -> 0 (keep)
    Clothing region  -> 255 (inpaint)
    """
    gray = cloth_image.convert("L")

    # Threshold: tweak 240–255 if needed for your dataset
    mask = gray.point(lambda p: 255 if p < 250 else 0)
    mask = mask.convert("L")
    return mask


# -----------------------------
# Lazy-loaded globals
# -----------------------------
CONTROLNET_PIPE: Optional[StableDiffusionControlNetInpaintPipeline] = None
OPENPOSE_DETECTOR: Optional[OpenposeDetector] = None


def _load_models():
    """
    Load ControlNet (OpenPose) and SD Inpaint pipeline, once.
    """
    global CONTROLNET_PIPE, OPENPOSE_DETECTOR

    if CONTROLNET_PIPE is not None and OPENPOSE_DETECTOR is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[ControlNet] Loading models on {device} with dtype={dtype}...")

    # 1) OpenPose ControlNet backbone
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=dtype,
    )

    # 2) Stable Diffusion Inpaint model with ControlNet
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=dtype,
    )

    # Optional: disable safety checker (for research use)
    def dummy_safety(images, **kwargs):
        return images, [False] * len(images)

    pipe.safety_checker = dummy_safety

    pipe.to(device)

    # 3) OpenPose detector (to turn person image into pose map)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    CONTROLNET_PIPE = pipe
    OPENPOSE_DETECTOR = openpose

    print("[ControlNet] Models loaded.")


# -----------------------------
# Main try-on function
# -----------------------------
def run_controlnet_tryon(
    person_image: Image.Image,
    cloth_image: Image.Image,
    mask_image: Optional[Image.Image] = None,
    pose_json: Optional[str] = None,  # not used
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
    controlnet_conditioning_scale: float = 1.0,
    prompt: str = "a realistic photo of the SAME person, wearing the garment",
) -> Image.Image:
    """
    Pose-aware virtual try-on using ControlNet OpenPose + SD Inpaint.

    Identity-preserving tweaks:
      - Only inpaint torso region (face & legs frozen)
      - Pre-paste cloth pattern on torso as a strong hint
    """

    _load_models()
    pipe = CONTROLNET_PIPE
    openpose = OPENPOSE_DETECTOR

    # ------------------- base images -------------------
    person_image = person_image.convert("RGB")
    cloth_image = cloth_image.convert("RGB")

    w, h = person_image.size

    # ---------- 1) compute a rough torso box ----------
    # tweak these ratios if needed
    top_y = int(0.20 * h)    # just below chin
    bottom_y = int(0.80 * h) # above hips / upper thighs
    left_x = int(0.20 * w)
    right_x = int(0.80 * w)

    torso_box = (left_x, top_y, right_x, bottom_y)

    # ---------- 2) create mask: white on torso, black elsewhere ----------
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle(torso_box, fill=255)

    # ---------- 3) paste cloth into torso region as a starting hint ----------
    base_image = person_image.copy()

    # resize cloth to fit torso box
    cloth_w = right_x - left_x
    cloth_h = bottom_y - top_y
    cloth_resized = cloth_image.resize((cloth_w, cloth_h))

    # optional: soften edges a bit
    cloth_resized = cloth_resized.filter(ImageFilter.GaussianBlur(radius=0.5))

    base_image.paste(cloth_resized, torso_box)

    # ---------- 4) generate OpenPose control image ----------
    pose_image = openpose(person_image).resize((w, h))

    # ---------- 5) run ControlNet + Inpaint ----------
    result = pipe(
        prompt=prompt,
        image=base_image,          # has cloth roughly pasted
        mask_image=mask,           # only torso is editable
        control_image=pose_image,  # pose constraint
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )

    return result.images[0]
