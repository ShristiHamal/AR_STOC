import os
import torch
import numpy as np
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline


# Optional: disable telemetry & xformers warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["XFORMERS_DISABLED"] = "1"


def generate_cloth_mask(cloth_image):
    """
    Auto-generate cloth mask from the clothing image.
    This mimics the dataset's cloth-mask (binary PNG).
    
    White background → 0 (don’t inpaint)
    Clothing region → 255 (inpaint)
    """

    # Convert cloth to grayscale
    gray = cloth_image.convert("L")

    # Threshold:
    # Pixels < 250 are clothing (foreground)
    # Pixels >= 250 are white background → masked out
    mask = gray.point(lambda p: 255 if p < 250 else 0)

    # Ensure mode L
    mask = mask.convert("L")
    return mask


def load_inpaint_model():
    """
    Load the Stable Diffusion Inpainting model on CPU or GPU.
    Called once for caching.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device)

    return pipe


# Cache the model (loaded only once)
PIPE = None


def run_inpaint_tryon(
    person_image,
    cloth_image,
    mask_image=None,
    pose_json=None,  # (ignored for inpainting, kept for compatibility)
    num_inference_steps=10,
    guidance_scale=5,
    prompt="a realistic virtual try-on of the person wearing the garment",
):
    """
    Perform virtual try-on using automatic cloth-mask.
    """

    global PIPE
    if PIPE is None:
        PIPE = load_inpaint_model()

    # ---------------------------------------
    # AUTO-MASK (dataset style cloth-mask)
    # ---------------------------------------
    if mask_image is None:
        mask_image = generate_cloth_mask(cloth_image)

    # Resize cloth mask to match person image
    mask_image = mask_image.resize(person_image.size)

    # Resize cloth to match person image
    cloth_image = cloth_image.resize(person_image.size)

    # ---------------------------------------
    # Run inpainting
    # ---------------------------------------
    result = PIPE(
        prompt=prompt,
        image=person_image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    out = result.images[0]
    return out
