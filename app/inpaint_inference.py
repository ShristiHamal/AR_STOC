import torch
from PIL import Image, ImageDraw
import numpy as np
from diffusers import StableDiffusionInpaintPipeline

# -----------------------------------------
# Device / dtype
# -----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# -----------------------------------------
# Load Stable Diffusion INPAINT pipeline
# -----------------------------------------
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=dtype,
).to(device)


# Disable NSFW filter safely
def dummy_safety_checker(images, **kwargs):
    # return images unchanged, and a list of "not nsfw" flags
    return images, [False] * len(images)


pipe.safety_checker = dummy_safety_checker


# -----------------------------------------
# Helper: simple rectangular torso mask
# -----------------------------------------
def make_torso_mask(size, top=0.25, bottom=0.80, left=0.20, right=0.80):
    """
    Creates a simple rectangular mask over the torso region.
    White (255) = area to inpaint (shirt).
    Black (0)   = keep as is.
    """
    W, H = size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    x1 = int(W * left)
    x2 = int(W * right)
    y1 = int(H * top)
    y2 = int(H * bottom)

    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask, (x1, y1, x2, y2)


# -----------------------------------------
# Main inpaint-based try-on function
# -----------------------------------------
def run_inpaint_tryon(
    person_image,
    cloth_image,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    prompt: str = "a realistic photo of the person wearing the pasted clothing",
):
    """
    person_image: PIL.Image (person)
    cloth_image:  PIL.Image (garment, front view)
    returns: PIL.Image with try-on result
    """

    # 1. Normalize inputs
    person_image = person_image.convert("RGB")
    cloth_image = cloth_image.convert("RGB")

    # Resize person to 512x512 (default SD size)
    base_size = 512
    person_image = person_image.resize((base_size, base_size), Image.BICUBIC)
    W, H = person_image.size

    # 2. Create torso mask + rectangle
    torso_mask, (x1, y1, x2, y2) = make_torso_mask((W, H))
    box_w, box_h = x2 - x1, y2 - y1

    # 3. Paste resized cloth into the torso area to create "coarse" try-on
    cloth_resized = cloth_image.resize((box_w, box_h), Image.BICUBIC)
    coarse_img = person_image.copy()
    coarse_img.paste(cloth_resized, (x1, y1))

    # 4. Run Stable Diffusion Inpainting
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = pipe(
                prompt=prompt,
                image=coarse_img,
                mask_image=torso_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
    else:
        
        out = pipe(
            prompt=prompt,
            image=coarse_img,
            mask_image=torso_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    return out
