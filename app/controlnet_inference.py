# app/controlnet_inference.py
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load models once
pose_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=pose_controlnet,
    torch_dtype=dtype
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = lambda images, **kwargs: (images, False)
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


def run_controlnet_inference(person_img_path, cloth_img_path, prompt="Virtual try-on", num_inference_steps=20):
    if not isinstance(person_img_path, Image.Image):
        person_img = Image.open(person_img_path).convert("RGB")
    else:
        person_img = person_img_path

    if not isinstance(cloth_img_path, Image.Image):
        cloth_img = Image.open(cloth_img_path).convert("RGB")
    else:
        cloth_img = cloth_img_path

    pose_img = pose_detector(person_img)

    with torch.autocast(device_type=device.type, dtype=dtype):
        result = pipe(prompt=prompt, image=pose_img, num_inference_steps=num_inference_steps).images[0]

    return result
