from clearml import PipelineDecorator, Logger
from pathlib import Path
import pandas as pd
import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image


# -------------------- Preprocessing -------------------- #
@PipelineDecorator.component(return_values=["df_path"])
def Preprocessing(dataset_txt_path: str):
    """
    Loads dataset mapping directly from train_pairs.txt.
    Expected format: each line contains image, cloth, cloth-mask, openpose_json paths.
    Saves CSV for downstream components.
    """
    dataset_txt_path = Path(dataset_txt_path)
    if not dataset_txt_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_txt_path}")

    try:
        df = pd.read_csv(dataset_txt_path, sep=None, engine="python", header=None)
    except Exception:
        df = pd.read_csv(dataset_txt_path, sep=r"\s+", engine="python", header=None)

    df.columns = ["image", "cloth", "cloth-mask", "openpose_json"]

    # Check for missing files
    missing = [p for col in df.columns for p in df[col] if not os.path.exists(p)]
    if missing:
        Logger.current_logger().report_text(f"Warning: {len(missing)} missing file paths detected.")
        print(f"Warning: {len(missing)} missing file paths. Example: {missing[:5]}")

    csv_path = dataset_txt_path.parent / "dataset_from_txt.csv"
    df.to_csv(csv_path, index=False)

    Logger.current_logger().report_table(
        title="Dataset Mapping (from train_pairs.txt)",
        series="train_data",
        table_plot=df.head(10)
    )
    print(f"Dataset loaded successfully: {len(df)} entries. CSV saved at {csv_path}")
    return str(csv_path)


# -------------------- ControlNet Inference -------------------- #
@PipelineDecorator.component(return_values=["processed_images_dir"])
def ControlNetInference(df_path: str, output_dir: str, sample_limit: int = None):
    """
    Runs ControlNet inference using OpenPose as conditioning.
    """
    df = pd.read_csv(df_path)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}, dtype: {dtype}")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    logger = Logger.current_logger()

    df_iter = df if sample_limit is None else df.head(sample_limit)
    for idx, row in df_iter.iterrows():
        try:
            person_img = Image.open(row["image"]).convert("RGB")
            pose = openpose(person_img)
            result = pipe(
                prompt=f"A person wearing {os.path.basename(row['cloth'])}",
                image=pose,
                num_inference_steps=20,
                guidance_scale=9.0,
            ).images[0]

            out_path = os.path.join(output_dir, f"result_{idx:05d}.png")
            result.save(out_path)
            logger.report_image(title="Try-On Result", series="results", local_path=out_path)
        except Exception as e:
            logger.report_text(f"Error processing {row['image']}: {e}")

    logger.report_text(f"Inference completed for {len(df_iter)} items. Results saved in {output_dir}")
    return output_dir


# -------------------- Training Component -------------------- #
@PipelineDecorator.component(return_values=["model_path"])
def Training(df_path: str):
    """
    Dummy training placeholder.
    """
    df = pd.read_csv(df_path)
    print(f"Training initialized with {len(df)} samples.")

    model_path = "./trained_model.pt"
    with open(model_path, "w") as f:
        f.write("fake model weights")

    Logger.current_logger().report_text("Training completed successfully.")
    return model_path


# -------------------- Full Pipeline -------------------- #
@PipelineDecorator.pipeline(
    name="AR Smart Try-On Full Pipeline",
    project="AR Smart Try-On",
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


# -------------------- Entry Point -------------------- #
if __name__ == "__main__":
    main_pipeline()
