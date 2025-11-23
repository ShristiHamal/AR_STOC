# app/inpaint_inference.py

import os

# Make sure diffusers does NOT try to use xformers
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

# -------------------------
# Device / dtype
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


# -------------------------
# Load Inpaint Pipeline
# -------------------------
# This is compatible with diffusers 0.21.0
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=dtype,
)
pipe = pipe.to(device)


def dummy_safety_checker(images, **kwargs):
    # return images unchanged and mark them as safe
    return images, [False] * len(images)


pipe.safety_checker = dummy_safety_checker


# -------------------------
# Mask helper
# -------------------------
def make_torso_mask(size, top=0.25, bottom=0.80, left=0.20, right=0.80):
    """
    Creates a simple rectangular mask over the torso region.
    White (255) = area to inpaint (shirt).
    Black (0)   = keep as is.
    """
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    x1 = int(w * left)
    x2 = int(w * right)
    y1 = int(h * top)
    y2 = int(h * bottom)

    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask, (x1, y1, x2, y2)


# -------------------------
# Main entry point
# -------------------------
def run_inpaint_tryon(
    person_image: Image.Image,
    cloth_image: Image.Image,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    prompt: str = "a realistic photo of the person wearing the pasted clothing",
) -> Image.Image:
    """
    Perform simple inpainting-based virtual try-on.

    - person_image: PIL.Image (person)
    - cloth_image:  PIL.Image (garment)
    - returns:      PIL.Image with try-on result
    """

    # Normalize
    person_image = person_image.convert("RGB")
    cloth_image = cloth_image.convert("RGB")

    # Resize person to SD default resolution
    base_size = 512
    person_image = person_image.resize((base_size, base_size), Image.BICUBIC)
    w, h = person_image.size

    # Torso mask and bounding box
    mask, (x1, y1, x2, y2) = make_torso_mask((w, h))
    box_w, box_h = x2 - x1, y2 - y1

    # Resize garment and paste on person
    cloth_resized = cloth_image.resize((box_w, box_h), Image.BICUBIC)
    coarse_img = person_image.copy()
    coarse_img.paste(cloth_resized, (x1, y1))

    # Inpainting step
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = pipe(
                prompt=prompt,
                image=coarse_img,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
    else:
        out = pipe(
            prompt=prompt,
            image=coarse_img,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    return out
