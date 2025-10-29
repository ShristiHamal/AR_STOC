# pipeline_decorator.py
from clearml import PipelineDecorator, Task, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import csv
import torch

# ClearML dataset ID
CLEARML_DATASET_ID = "8832df278eb245b2856da6c202aaa876"

@PipelineDecorator.pipeline(
    name="AR_TryOn",
    project="AR_STOC",
    version="1.0.0",
    pipeline_execution_queue="ar_stoc",
    
)
def full_pipeline(output_dir: str):
    from clearml import Task

    task = Task.init(
    project_name="AR_TryOn",
    task_name="AR_Pipeline",
    repo="https://github.com/ShristiHamal/AR_STOC.git"
)

    # Task.current_task().connect({"output_dir": output_dir})

    # Fetch dataset from ClearML
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    root_dir = dataset.get_local_copy()

    csv_path = preprocessing(root_dir)
    inference_dir = training(csv_path, output_dir)
    metrics = evaluation(inference_dir)
    return metrics


@PipelineDecorator.component(name="preprocessing", return_values=["csv_path"])
def preprocessing(root_dir: str) -> str:
    """
    Preprocess images and masks and save as CSV
    """
    root = Path(root_dir)
    # Load the npy files from ClearML dataset
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

    return str(csv_file)


@PipelineDecorator.component(name="training", return_values=["inference_dir"])
def training(csv_path: str, output_dir: str) -> str:
    """
    Run ControlNet inference to generate virtual try-on images
    """
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from controlnet_aux import OpenposeDetector

    # IMPORTANT: The 'device' here will be the device available on the ClearML Agent
    # If your agent has a GPU, it will use 'cuda'. If not, it will fall back to 'cpu'.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    task = Task.current_task()
    logger = task.get_logger()

    # Load ControlNet models - These will be downloaded by the agent
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
                num_inference_steps=8
            ).images[0]
        return result

    with Path(csv_path).open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

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

    return str(output_path)


@PipelineDecorator.component(name="evaluation", return_values=["metrics"])
def evaluation(inference_dir: str) -> dict:
    """
    Compute simple image metrics for generated images
    """
    output_path = Path(inference_dir)
    scores = []
    for img_file in output_path.glob("tryon_*.png"):
        try:
            img = Image.open(img_file).convert("RGB")
            arr = np.array(img)
            scores.append({
                "file": img_file.name,
                "mean_pixel": float(arr.mean()),
                "std_pixel": float(arr.std())
            })
        except Exception as e:
            print(f"Failed to evaluate {img_file.name}: {e}")
    print(f"Evaluated {len(scores)} images")
    return {"image_stats": scores}


if __name__ == "__main__":
    pipeline_instance = full_pipeline(
        output_dir=r"C:/Users/shris/OneDrive - UTS/Documents/GitHub/AR_STOC/controlnet_outputs",
    )

    # Explicitly set the repo for the agent
    pipeline_instance.start(
        queue="ar_stoc",
        repo="https://github.com/ShristiHamal/AR_STOC.git",
        branch="main" 
    )
    