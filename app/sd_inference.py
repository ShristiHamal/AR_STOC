import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=False
).to(device)

pipe.safety_checker = lambda images, **kwargs: (images, False)  # Optional: disable NSFW filter

def run_sd_inference(person_img_path, cloth_img_path, mask_img_path, output_dir, prompt="Virtual try-on"):
    try:
        person_img = Image.open(person_img_path).convert("RGB")
        mask_img = Image.open(mask_img_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    if not isinstance(person_img, Image.Image) or not isinstance(mask_img, Image.Image):
        raise TypeError("Inputs must be PIL.Image objects")

    print(f"[DEBUG] person_img type: {type(person_img)}, mask_img type: {type(mask_img)}")

    try:
        result = pipe(prompt=prompt, image=person_img, mask_image=mask_img).images[0]
    except Exception as e:
        raise RuntimeError(f"Inpainting failed for {person_img_path}: {e}")

    return result