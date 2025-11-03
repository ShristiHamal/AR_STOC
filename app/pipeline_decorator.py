# app/pipeline_decorator.py

from clearml import PipelineDecorator, Logger
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import os
import pandas as pd
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from contextlib import nullcontext

# -------------------- Preprocessing -------------------- #
@PipelineDecorator.component(return_values=["df"])
def preprocess_data(dataset_root: str):
    """
    Preprocess dataset: maps person images, cloth images, masks, and pose JSON.
    Returns a DataFrame for downstream components.
    """
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "image"
    cloth_dir = dataset_root / "cloth"
    mask_dir = dataset_root / "cloth-mask"
    openpose_dir = dataset_root / "openpose_json"

    image_files = sorted(image_dir.glob("*.jpg"))[:200]  # adjust as needed
    data = []

    for img_file in image_files:
        base_name = "_".join(img_file.stem.split("_")[:2])  # e.g., "00001_00"

        cloth_match = cloth_dir / f"{base_name}.jpg"
        mask_match = mask_dir / f"{base_name}.jpg"
        pose_match = openpose_dir / f"{base_name}_keypoints.json"

        if cloth_match.exists() and mask_match.exists() and pose_match.exists():
            data.append({
                "person_image": str(img_file),
                "cloth_image": str(cloth_match),
                "mask_image": str(mask_match),
                "pose_json": str(pose_match)
            })

    df = pd.DataFrame(data)
    Logger.current_logger().report_table(
        title="Dataset Mapping",
        series="train_data",
        table_plot=df.head(10)
    )
    print(f"Preprocessing done: {len(df)} valid pairs found.")
    return df

# -------------------- Inference -------------------- #
@PipelineDecorator.component(return_values=["output_dir"])
def run_controlnet_inference(df, output_dir: str):
    """
    Run ControlNet inference using Stable Diffusion + OpenPose.
    Saves results in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load models
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    torch.manual_seed(42)
    np.random.seed(42)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" else nullcontext()
    )

    for idx, row in df.iterrows():  # full dataset
        person_img = Image.open(row["person_image"]).convert("RGB")
        cloth_img = Image.open(row["cloth_image"]).convert("RGB")
        mask_img = Image.open(row["mask_image"]).convert("RGB")

        # Optionally, you can load the pose from JSON instead of detecting
        pose_img = pose_detector(person_img)
        if isinstance(pose_img, np.ndarray):
            pose_img = Image.fromarray(pose_img)

        with autocast_ctx:
            result = pipe(
                prompt="Person wearing the selected clothing item in realistic style",
                control_image=[pose_img, mask_img],
                num_inference_steps=8
            ).images[0]

        output_path = os.path.join(output_dir, f"result_{idx:04d}.png")
        result.save(output_path)
        Logger.current_logger().report_image(
            title="AR Try-On Result",
            series="results",
            local_path=output_path
        )

    print(f"Inference done. Results saved in {output_dir}")
    return output_dir

# -------------------- Evaluation -------------------- #
@PipelineDecorator.component()
def evaluate_results(output_dir: str):
    """
    Evaluate generated images. Placeholder for metrics or visual checks.
    """
    generated_files = list(Path(output_dir).glob("*.png"))
    Logger.current_logger().report_text(f"Generated {len(generated_files)} try-on images.")
    print(f"Evaluation complete: {len(generated_files)} results generated.")

# -------------------- Full Pipeline -------------------- #
@PipelineDecorator.pipeline(
    name="AR Smart Try-On Full Pipeline",
    project="AR Smart Try-On",
    version="1.0.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc"
)
def full_pipeline(dataset_root: str, output_dir: str):
    """
    Pipeline:
    1. Preprocess dataset
    2. Run ControlNet inference
    3. Evaluate results
    """
    df = preprocess_data(dataset_root)
    results_dir = run_controlnet_inference(df, output_dir)
    evaluate_results(results_dir)

# -------------------- Entry Point -------------------- #
if __name__ == "__main__":
    full_pipeline(
        dataset_root="/content/drive/MyDrive/IndustryProject/Dataset/train",
        output_dir="/content/drive/MyDrive/IndustryProject/Results"
    )
