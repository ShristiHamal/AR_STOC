# app/pipeline_decorator.py
from clearml import PipelineDecorator, Dataset, Task
from pathlib import Path
from PIL import Image
import numpy as np
import csv
import torch

# Dataset ID
CLEARML_DATASET_ID = "8832df278eb245b2856da6c202aaa876"

@PipelineDecorator.pipeline(
    name="AR_TryOn_Pipeline",
    project="AR_STOC",
    version="1.0.0",
    pipeline_execution_queue="ar_stoc",
)
def full_pipeline(output_dir: str):
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    root_dir = dataset.get_local_copy()

    csv_path = preprocessing(root_dir)
    inference_dir = training(csv_path, output_dir)
    metrics = evaluation(inference_dir)
    return metrics


@PipelineDecorator.component(return_values=["csv_path"])
def preprocessing(root_dir: str) -> str:
    root = Path(root_dir)
    images = np.load(root / "image.npy")
    masks = np.load(root / "cloth_mask.npy")

    csv_file = root / "dataset.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["person_image", "mask_image"])
        writer.writeheader()
        for i in range(len(images)):
            person_img_path = root / f"img_{i}.png"
            mask_img_path = root / f"mask_{i}.png"
            Image.fromarray(images[i]).save(person_img_path)
            Image.fromarray(masks[i]).save(mask_img_path)
            writer.writerow({
                "person_image": str(person_img_path),
                "mask_image": str(mask_img_path)
            })

    print(f" Preprocessing complete. Saved CSV at {csv_file}")
    return str(csv_file)


@PipelineDecorator.component(return_values=["inference_dir"])
def training(csv_path: str, output_dir: str) -> str:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from controlnet_aux import OpenposeDetector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"🧩 Using device: {device}")

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

    def run_inference(person_img_path, mask_img_path, prompt="Virtual try-on"):
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
                num_inference_steps=8
            ).images[0]
        return result

    with Path(csv_path).open(newline='') as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, start=1):
        try:
            result = run_inference(row['person_image'], row['mask_image'])
            out_path = output_path / f"tryon_{i}.png"
            result.save(out_path)
            print(f"[{i}] ✅ Generated {out_path}")
        except Exception as e:
            print(f"[{i}] Failed: {e}")

    return str(output_path)


@PipelineDecorator.component(return_values=["metrics"])
def evaluation(inference_dir: str) -> dict:
    output_path = Path(inference_dir)
    scores = []
    for img_file in output_path.glob("tryon_*.png"):
        try:
            arr = np.array(Image.open(img_file).convert("RGB"))
            scores.append({
                "file": img_file.name,
                "mean_pixel": float(arr.mean()),
                "std_pixel": float(arr.std())
            })
        except Exception as e:
            print(f"Failed to evaluate {img_file.name}: {e}")
    print(f" Evaluation complete for {len(scores)} images.")
    return {"image_stats": scores}
