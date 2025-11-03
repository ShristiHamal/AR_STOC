from clearml import PipelineDecorator, Task, Logger, Dataset
from pathlib import Path
import pandas as pd
import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from contextlib import nullcontext
from PIL import Image
import numpy as np


# -------------------- Preprocessing -------------------- #
@PipelineDecorator.component(return_values=["df_path"])
def Preprocessing(dataset_txt_path: str):
    """
    Loads dataset mapping directly from an existing train_pairs.txt file.
    Expected format: each line contains image, cloth, cloth-mask, openpose_json paths separated by spaces or commas.
    Saves a DataFrame to CSV for easier downstream use.
    """
    dataset_txt_path = Path(dataset_txt_path)
    if not dataset_txt_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_txt_path}")

    # Try loading .txt file as CSV with auto delimiter detection
    try:
        df = pd.read_csv(dataset_txt_path, sep=None, engine="python", header=None)
    except Exception:
        # fallback to whitespace split
        df = pd.read_csv(dataset_txt_path, sep=r"\s+", engine="python", header=None)

    # Assign column names for consistency
    df.columns = ["image", "cloth", "cloth-mask", "openpose_json"]

    # Verify all referenced files exist
    missing = []
    for col in df.columns:
        for p in df[col]:
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        Logger.current_logger().report_text(f"Warning: {len(missing)} missing file paths detected.")
        print(f"Warning: {len(missing)} missing file paths detected. Example: {missing[:5]}")

    # Save CSV for downstream components
    csv_path = dataset_txt_path.parent / "dataset_from_txt.csv"
    df.to_csv(csv_path, index=False)

    Logger.current_logger().report_table(
        title="Dataset Mapping (from train_pairs.txt)",
        series="train_data",
        table_plot=df.head(10)
    )
    print(f"Dataset loaded successfully: {len(df)} entries found. CSV saved at {csv_path}")
    return str(csv_path)


# -------------------- ControlNet Inference -------------------- #
@PipelineDecorator.component(return_values=["processed_images_dir"])
def ControlNetInference(df_path: str, output_dir: str):
    """
    Runs ControlNet inference using OpenPose as conditioning.
    """
    df = pd.read_csv(df_path)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading OpenPose and ControlNet models...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    for idx, row in df.iterrows():
        person_image_path = row["image"]
        cloth_image_path = row["cloth"]

        try:
            person_img = Image.open(person_image_path).convert("RGB")
            pose = openpose(person_img)
            result = pipe(
                prompt=f"A person wearing {os.path.basename(cloth_image_path)}",
                image=pose,
                num_inference_steps=20,
                guidance_scale=9.0,
            ).images[0]

            out_path = os.path.join(output_dir, f"result_{idx:05d}.png")
            result.save(out_path)
        except Exception as e:
            Logger.current_logger().report_text(f"Error processing {person_image_path}: {e}")

    Logger.current_logger().report_text(f"Inference completed for {len(df)} items. Results in {output_dir}")
    return output_dir


# -------------------- Training Component -------------------- #
@PipelineDecorator.component(return_values=["model_path"])
def Training(df_path: str):
    """
    Dummy Training step placeholder — load data, simulate training, save model.
    """
    df = pd.read_csv(df_path)
    print(f"Training initialized with {len(df)} samples.")

    # Simulate training
    model_path = "./trained_model.pt"
    with open(model_path, "w") as f:
        f.write("fake model weights")

    Logger.current_logger().report_text("Training completed successfully.")
    return model_path


# -------------------- Pipeline Definition -------------------- #
@PipelineDecorator.pipeline(
    name="AR-STOC ControlNet Pipeline",
    project="AR_STOC",
    version="1.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc"
    
)
def main_pipeline():
    dataset_txt_path = "/content/drive/MyDrive/IndustryProject/Dataset/train_pairs.txt"

    df_path = Preprocessing(dataset_txt_path)
    processed_images_dir = ControlNetInference(df_path, output_dir="./results")
    model_path = Training(df_path)

    print("Pipeline completed successfully.")
    return model_path


if __name__ == "__main__":
    PipelineDecorator.run_pipeline(main_pipeline)
