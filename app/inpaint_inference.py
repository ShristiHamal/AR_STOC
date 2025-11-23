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
    person_image: Image.Image,
    cloth_image: Image.Image,
    mask_image: Image.Image,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    prompt: str = "a realistic photo of the person wearing the garment",
) -> Image.Image:
    """
    person_image: PIL.Image (RGB) - person wearing original clothes
    cloth_image:  PIL.Image (RGB) - front-view garment
    mask_image:   PIL.Image (L or 1) - binary mask of the garment (same cloth space)
    returns: PIL.Image with try-on result
    """

    # 1. Normalize inputs
    person_image = person_image.convert("RGB")
    cloth_image = cloth_image.convert("RGB")
    mask_image = mask_image.convert("L")

    # Stable Diffusion inpainting works best around 512x512
    base_size = 512
    person_image = person_image.resize((base_size, base_size), Image.BICUBIC)
    W, H = person_image.size

    # 2. Compute torso box where we want the garment
    x1, y1, x2, y2 = _torso_box(W, H)
    box_w, box_h = x2 - x1, y2 - y1

    # 3. Resize cloth & mask to that box
    cloth_resized = cloth_image.resize((box_w, box_h), Image.BICUBIC)
    mask_resized = mask_image.resize((box_w, box_h), Image.NEAREST)

    # 4. Create a full-size inpaint mask (0 = keep, 255 = inpaint)
    inpaint_mask = Image.new("L", (W, H), 0)
    inpaint_mask.paste(mask_resized, (x1, y1))

    # 5. Create coarse image: paste garment into person using alpha from mask
    coarse_img = person_image.copy()
    # Convert mask_resized to proper alpha
    cloth_rgba = cloth_resized.copy()
    cloth_rgba.putalpha(mask_resized)
    coarse_img.paste(cloth_rgba, (x1, y1), mask_resized)

    # 6. Run Stable Diffusion inpainting
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = pipe(
                prompt=prompt,
                image=coarse_img,
                mask_image=inpaint_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
    else:
        out = pipe(
            prompt=prompt,
            image=coarse_img,
            mask_image=inpaint_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    return out
