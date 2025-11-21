import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from controlnet_aux import OpenposeDetector

# ------------------------------------------------------------
# Device / dtype setup
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


# ------------------------------------------------------------
# Load ControlNet + Stable Diffusion pipeline
# ------------------------------------------------------------
pose_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=dtype
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=pose_controlnet,
    dtype=dtype
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# FIX: Correct safety checker to avoid `'bool' is not iterable`
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker

# Load OpenPose for auto pose extraction
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


# ------------------------------------------------------------
# Virtual Try-On Inference Function
# ------------------------------------------------------------
def run_controlnet_inference(
    person_image,
    cloth_image,
    num_inference_steps=20,
    strength=0.65,
    prompt="A person wearing the given clothing, photorealistic",
):
    """
    Inputs:
        person_image: PIL Image (RGB)
        cloth_image: PIL Image (RGB)
    Output:
        PIL Image result from Stable Diffusion + ControlNet refinement
    """

    # Normalize both images to RGB
    person_image = person_image.convert("RGB")
    cloth_image = cloth_image.convert("RGB")

    W, H = person_image.size

    # --------------------------------------------------------
    # Step 1 — AUTO-GENERATE POSE MAP
    # --------------------------------------------------------
    pose_map = pose_detector(person_image)
    pose_map = pose_map.convert("RGB").resize((W, H), Image.BICUBIC)

    # --------------------------------------------------------
    # Step 2 — Resize clothing to upper torso size
    # --------------------------------------------------------
    new_w = int(W * 0.45)
    new_h = int(H * 0.45)
    cloth_resized = cloth_image.resize((new_w, new_h), Image.BICUBIC)

    # Convert to numpy arrays
    cloth_np = np.array(cloth_resized)
    person_np = np.array(person_image.copy())

    # --------------------------------------------------------
    # Step 3 — Paste clothing onto torso to create coarse image
    # --------------------------------------------------------
    x = W // 2 - new_w // 2     # center horizontally
    y = int(H * 0.32)           # place on upper torso

    # Ensure safe placement
    if 0 <= x < W and 0 <= y < H and (y + new_h) <= H and (x + new_w) <= W:
        person_np[y:y + new_h, x:x + new_w] = cloth_np

    coarse_img = Image.fromarray(person_np)

    # Ensure coarse image is exactly same size as pose map
    coarse_img = coarse_img.resize((W, H), Image.BICUBIC)

    # --------------------------------------------------------
    # Step 4 — Run Stable Diffusion with ControlNet refinement
    # --------------------------------------------------------
    with torch.autocast(device_type=device, dtype=dtype if device == "cuda" else torch.float32):
        result = pipe(
            prompt=prompt,
            image=coarse_img,
            control_image=pose_map,
            num_inference_steps=num_inference_steps,
            strength=strength,
        ).images[0]

    return result
