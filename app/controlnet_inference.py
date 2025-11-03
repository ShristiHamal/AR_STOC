from PIL import Image
import torch
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector


# -------------------- Initialization -------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load ControlNet models
pose_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype
)
seg_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=dtype
)

# Load Stable Diffusion + ControlNet pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[pose_controlnet, seg_controlnet],
    torch_dtype=dtype
).to(device)

# Disable NSFW filter for inference
pipe.safety_checker = lambda images, **kwargs: (images, False)

# Initialize pose detector
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


# -------------------- Inference Function -------------------- #
def run_controlnet_inference(person_img_path, mask_img_path, prompt="Virtual try-on", num_inference_steps=8):
    """
    Generates a ControlNet-guided image using a person image and segmentation mask.
    """

    # Load and resize input images
    if not isinstance(person_img_path, Image.Image):
        person_img = Image.open(person_img_path).convert("RGB").resize((512, 512))
    else:
        person_img = person_img_path.resize((512, 512))

    mask_img = Image.open(mask_img_path).convert("RGB").resize((512, 512))

    # Detect human pose
    pose_img = pose_detector(person_img)
    if isinstance(pose_img, np.ndarray):
        pose_img = Image.fromarray(pose_img)

    # Run ControlNet inference
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype):
        result = pipe(
            prompt=prompt,
            image=person_img,
            control_image=[pose_img, mask_img],
            num_inference_steps=num_inference_steps
        ).images[0]

    return result
