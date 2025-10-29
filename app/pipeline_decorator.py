# pipeline_decorator.py

from clearml import PipelineDecorator, Task, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import csv
import torch

# ----------------------------------------------------
# Full Pipeline Definition
# ----------------------------------------------------
@PipelineDecorator.pipeline(
    name="AR_TryOn_Full",
    project="AR_STOC",
    version="1.0.0",
    default_queue="ar_stoc",            # Component default queue
    pipeline_execution_queue="ar_stoc"  # Main controller queue
)
def full_pipeline(output_dir: str):
    """
    Full AR Try-On pipeline: preprocessing -> training/inference -> evaluation
    """
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_text("Starting full AR Try-On pipeline")

    # Step 1: Preprocessing
    csv_path = preprocessing()

    # Step 2: Training / Inference
    inference_dir = training(csv_path, output_dir)

    # Step 3: Evaluation
    metrics = evaluation(inference_dir)

    logger.report_text("Pipeline completed successfully")
    return metrics


# ----------------------------------------------------
# Components
# ----------------------------------------------------
@PipelineDecorator.component(
    name="preprocessing",
    return_values=["csv_path"],
    cache=True,
    task_type=Task.TaskTypes.data_processing,
)
def preprocessing(
    CLEARML_DATASET_ID: str = "c1fca92f4cc1402fac5fd6026c1128e5",
    CLEARML_DATASET_NAME: str = "AR_TryOn_Train"
) -> str:
    """
    Fetch ClearML dataset, convert numpy arrays to PNGs, create CSV for inference.
    """
    task = Task.current_task()
    logger = task.get_logger()

    # Fetch dataset from ClearML
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    root = Path(dataset.get_local_copy())
    logger.report_text(f"Dataset local path: {root}")

    # Load numpy arrays
    cloth_images = np.load(root / "train_cloth.npy")
    cloth_masks = np.load(root / "train_cloth_mask.npy")
    person_images = np.load(root / "train_image.npy")

    # Create CSV file mapping all images
    csv_file = root / "dataset.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["person_image", "mask_image", "cloth_image"])
        writer.writeheader()

        for i in range(len(cloth_images)):
            person_img_path = root / f"person_{i}.png"
            mask_img_path = root / f"mask_{i}.png"
            cloth_img_path = root / f"cloth_{i}.png"

            Image.fromarray(person_images[i]).save(person_img_path)
            Image.fromarray(cloth_masks[i]).save(mask_img_path)
            Image.fromarray(cloth_images[i]).save(cloth_img_path)

            writer.writerow({
                "person_image": str(person_img_path),
                "mask_image": str(mask_img_path),
                "cloth_image": str(cloth_img_path)
            })

    logger.report_text(f"Preprocessing completed. CSV saved at: {csv_file}")
    return str(csv_file)


@PipelineDecorator.component(
    name="training",
    return_values=["inference_dir"],
    cache=False,
    task_type=Task.TaskTypes.training,
)
def training(csv_path: str, output_dir: str) -> str:
    """
    Run ControlNet inference to generate virtual try-on images.
    """
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from controlnet_aux import OpenposeDetector

    task = Task.current_task()
    logger = task.get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.report_text("Initializing ControlNet models...")

    # Load ControlNet models
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.report_text(f"Processing {len(rows)} image pairs...")

    for idx, row in enumerate(rows, start=1):
        try:
            person_img = Image.open(row['person_image']).convert("RGB").resize((512, 512))
            mask_img = Image.open(row['mask_image']).convert("RGB").resize((512, 512))
            cloth_img = Image.open(row['cloth_image']).convert("RGB").resize((512, 512))

            pose_img = pose_detector(person_img)
            if isinstance(pose_img, np.ndarray):
                pose_img = Image.fromarray(pose_img)

            # Inference
            with torch.autocast(device_type=device.type, dtype=dtype):
                result = pipe(
                    prompt="Virtual try-on of the selected clothing item",
                    image=person_img,
                    control_image=[pose_img, mask_img],
                    num_inference_steps=8
                ).images[0]

            out_file = output_path / f"tryon_{Path(row['person_image']).name}"
            result.save(out_file)

            logger.report_image(
                title="Try-On Result",
                series="ControlNet Output",
                iteration=idx,
                image=str(out_file)
            )
            logger.report_text(f"[{idx}/{len(rows)}] Processed {row['person_image']}")

        except Exception as e:
            logger.report_text(f"[{idx}] Failed: {e}")

    logger.report_text(f"Inference completed. Results saved in {output_path}")
    return str(output_path)


@PipelineDecorator.component(
    name="evaluation",
    return_values=["metrics"],
    cache=False,
    task_type=Task.TaskTypes.testing,
)
def evaluation(inference_dir: str) -> dict:
    """
    Compute simple evaluation metrics for generated try-on images.
    """
    task = Task.current_task()
    logger = task.get_logger()
    output_path = Path(inference_dir)

    metrics = []
    for img_file in output_path.glob("tryon_*.png"):
        try:
            img = Image.open(img_file).convert("RGB")
            arr = np.array(img)
            metrics.append({
                "file": img_file.name,
                "mean_pixel": float(arr.mean()),
                "std_pixel": float(arr.std())
            })
        except Exception as e:
            logger.report_text(f"Failed to evaluate {img_file.name}: {e}")

    logger.report_text(f"Evaluation completed. {len(metrics)} images processed.")
    return {"image_metrics": metrics}


# ----------------------------------------------------
# Pipeline Entry Point
# ----------------------------------------------------
if __name__ == "__main__":
    pipeline_instance = full_pipeline(
        output_dir=r"C:/Users/shris/OneDrive - UTS/Documents/GitHub/AR_STOC/controlnet_outputs"
    )

    pipeline_instance.start(
        repo="https://github.com/ShristiHamal/AR_STOC.git",
        branch="main"
    )
