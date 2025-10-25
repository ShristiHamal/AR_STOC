import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load models ONCE
pose_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype
)
seg_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=dtype
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[pose_controlnet, seg_controlnet],
    torch_dtype=dtype
).to(device)
pipe.safety_checker = lambda images, **kwargs: (images, False)
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

@torch.inference_mode()
def run_controlnet_inference(person_img_path, mask_img_path, prompt="Virtual try-on"):
    person_img = Image.open(person_img_path).convert("RGB").resize((512, 512))
    mask_img = Image.open(mask_img_path).convert("RGB").resize((512, 512))
    pose_img = pose_detector(person_img)
    if isinstance(pose_img, np.ndarray):
        pose_img = Image.fromarray(pose_img)

    with torch.autocast(device_type=device.type, dtype=dtype):
        result = pipe(
            prompt=prompt,
            image=person_img,
            control_image=[pose_img, mask_img],
            num_inference_steps=8  # was 20
        ).images[0]

    result.save("debug_output.jpg")
    return result

def batch_infer(rows, output_path, logger):
    for row_num, row in enumerate(rows, start=1):
        try:
            person_path = row['person_image']
            mask_path = row['mask_image']
            stem = Path(person_path).name

            result = run_controlnet_inference(person_path, mask_path)
            out_path = output_path / f"tryon_{stem}"
            result.save(out_path)

            logger.report_image("Try-On Result", "Dual ControlNet", row_num, str(out_path))
            print(f"[Row {row_num}] Processed {stem}")

        except Exception as e:
            print(f"[Row {row_num}] Failed on {row.get('person_image', 'unknown')}: {e}")
