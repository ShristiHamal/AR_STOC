import torch
import numpy as np
import json
import cv2
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from controlnet_aux import OpenposeDetector

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


# Load ControlNet model and Stable Diffusion pipeline
pose_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=dtype
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[pose_controlnet],
    torch_dtype=dtype
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = lambda images, **kwargs: (images, False)

pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


#Convert OpenPose JSON to pose map
def pose_json_to_map(json_path, width, height):
    """Convert OpenPose JSON keypoints into ControlNet pose map."""
    with open(json_path, "r") as f:
        data = json.load(f)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if "people" not in data or len(data["people"]) == 0:
        return Image.fromarray(canvas)

    pts = data["people"][0].get("pose_keypoints_2d", [])

    for i in range(0, len(pts), 3):
        x, y, conf = pts[i:i + 3]
        if x > 0 and y > 0:
            cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), -1)

    return Image.fromarray(canvas)


# Clothes wrapping based on torso size
def tps_warp_cloth(cloth, mask, target_h, target_w):
    """Resizes cloth roughly to body width using mask."""
    cloth = cloth.convert("RGBA")
    mask = mask.resize(cloth.size, Image.NEAREST)

    cloth_np = np.array(cloth)
    mask_np = np.array(mask)

    # Apply mask to cloth
    garment = cv2.bitwise_and(cloth_np, cloth_np, mask=mask_np)

    # Resize garment to 45% of torso width (tunable)
    new_w = int(target_w * 0.45)
    new_h = int(target_h * 0.45)

    if new_w <= 0 or new_h <= 0:
        return garment

    garment_resized = cv2.resize(garment, (new_w, new_h))

    return garment_resized


#Virtual Try-On Inference Function
def run_controlnet_inference(
    person_img_path,
    cloth_img_path,
    mask_img_path,
    pose_json_path,
    prompt="A person wearing the given clothing, photorealistic",
    num_inference_steps=20,
    strength=0.65
):
#Load images
    person = Image.open(person_img_path).convert("RGB")
    cloth = Image.open(cloth_img_path).convert("RGB")
    mask = Image.open(mask_img_path).convert("L")

    W, H = person.size

#Generate pose map from JSON
    pose_map = pose_json_to_map(pose_json_path, W, H)

    # ---------------------------
    # 3. Warp cloth
    # ---------------------------
    warped = tps_warp_cloth(cloth, mask, H, W)
    h, w = warped.shape[:2]

# Create coarse composite
    coarse = np.array(person.copy())

    # Position the garment
    x = W // 2 - w // 2     # place in horizontal center
    y = int(H * 0.32)       # place roughly on upper torso

    # Safe bounds
    if y + h <= H and x + w <= W and x >= 0 and y >= 0:
        coarse[y:y+h, x:x+w] = warped

    coarse_img = Image.fromarray(coarse)

#Stable Diffusion Inference
    with torch.autocast(device_type=device, dtype=dtype):
        result = pipe(
            prompt=prompt,
            image=coarse_img,
            control_image=pose_map,
            num_inference_steps=num_inference_steps,
            strength=strength
        ).images[0]

    return result
