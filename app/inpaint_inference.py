# app/inpaint_inference.py

import os
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


# ------------------------------------------------------------
# Device + dtype
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


# ------------------------------------------------------------
# Load Stable Diffusion Inpaint Pipeline
# Works with diffusers<=0.19.3 and CPU
# ------------------------------------------------------------
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",                     # avoids xformers crash
    torch_dtype=dtype,
)
pipe.to(device)

pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))


# ------------------------------------------------------------
# Coarse Try-On + Inpaint refinement
# ------------------------------------------------------------
def run_inpaint_tryon(
    person_image,
    cloth_image,
    mask_image,
    pose_json=None,
    num_inference_steps=25,
    guidance_scale=7.5,
    prompt="a realistic virtual try-on of the person wearing the garment",
):
    """
    Inputs:
        person_image (PIL.Image)
        cloth_image  (PIL.Image)
        mask_image   (PIL.Image, L mode) – white=replace, black=keep
        pose_json    (dict) OPTIONAL
    """

    # --- Normalize inputs
    person = person_image.convert("RGB").resize((512, 512))
    cloth  = cloth_image.convert("RGB").resize((512, 512))
    mask   = mask_image.convert("L").resize((512, 512))

    # --- Composite for better conditioning
    coarse = person.copy()
    coarse.paste(cloth, (0, 0), mask)

    # --- Inpaint refinement
    if device == "cuda":
        with torch.autocast("cuda"):
            out = pipe(
                prompt=prompt,
                image=coarse,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
    else:
        out = pipe(
            prompt=prompt,
            image=coarse,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    return out
