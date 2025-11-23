# app/inpaint_inference.py

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

# -------------------------
# Device / dtype
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


# -------------------------
# Load Stable Diffusion INPAINT pipeline
# -------------------------
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=dtype,
)
pipe = pipe.to(device)


def _dummy_safety_checker(images, **kwargs):
    # Return images unchanged and "not nsfw" flags
    return images, [False] * len(images)


pipe.safety_checker = _dummy_safety_checker


# -------------------------
# Helper: torso box (for placement)
# -------------------------
def _torso_box(width: int, height: int):
    """
    Returns a heuristic torso bounding box in (x1, y1, x2, y2)
    coordinates, relative to a 512x512-ish portrait.
    """
    top = 0.25
    bottom = 0.80
    left = 0.20
    right = 0.80
    x1 = int(width * left)
    x2 = int(width * right)
    y1 = int(height * top)
    y2 = int(height * bottom)
    return x1, y1, x2, y2


# -------------------------
# Main inpaint-based try-on
# -------------------------
def run_inpaint_tryon(
    person_image,
    cloth_image,
    mask_image,
    pose_json=None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    prompt: str = "a realistic photo of the person wearing the pasted clothing",
):
    """
    Inpainting-based try-on using a pre-made binary mask.
    """

    # Ensure correct modes
    person_image = person_image.convert("RGB")
    cloth_image  = cloth_image.convert("RGB")
    mask_image   = mask_image.convert("L")  # grayscale

    # Resize inputs to 512x512 (Stable Diffusion default)
    W, H = 512, 512
    person_resized = person_image.resize((W, H))
    mask_resized   = mask_image.resize((W, H))
    cloth_resized  = cloth_image.resize((W, H))

    # Insert rough cloth into masked region
    coarse = person_resized.copy()
    coarse.paste(cloth_resized, (0, 0), mask_resized)

    # Run inpainting
    if torch.cuda.is_available():
        with torch.autocast("cuda"):
            out = pipe(
                prompt=prompt,
                image=coarse,
                mask_image=mask_resized,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
    else:
        out = pipe(
            prompt=prompt,
            image=coarse,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    return out
