from clearml import PipelineDecorator, Logger
from pathlib import Path
import pandas as pd
import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np

# -------------------- ControlNet Inference -------------------- #
@PipelineDecorator.component(return_values=["processed_images_dir"])
def ControlNetInference(df_path: str, output_dir: str, sample_limit: int = None):
    """
    Runs ControlNet inference using OpenPose conditioning.
    df_path: Path to train_pairs.txt or CSV listing images, cloth, mask, pose JSON.
    """
    # Try auto-detect separator in CSV or txt
    df = pd.read_csv(df_path, sep=None, engine="python", header=None)
    if df.shape[1] == 4:  # If 4 columns, assign names
        df.columns = ["image", "cloth", "cloth-mask", "openpose_json"]

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger = Logger.current_logger()
    logger.report_text(f"Using device: {device}, dtype: {dtype}")

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
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
    Dummy training step (placeholder)
    """
    df = pd.read_csv(df_path, sep=None, engine="python", header=None)
    print(f"Training initialized with {len(df)} samples.")
    model_path = "./trained_model.pt"
    with open(model_path, "w") as f:
        f.write("fake model weights")
    Logger.current_logger().report_text("Training completed successfully.")
    return model_path

# -------------------- Evaluation Component -------------------- #
@PipelineDecorator.component(return_values=["metrics"])
def Evaluation(processed_images_dir: str):
    """
    Basic evaluation: computes mean and std pixel values for generated images.
    """
    image_files = list(Path(processed_images_dir).glob("*.png"))
    results = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        results.append({
            "image": img_path.name,
            "mean_pixel": float(arr.mean()),
            "std_pixel": float(arr.std())
        })
    Logger.current_logger().report_text(f"Evaluation completed for {len(results)} images.")
    return {"metrics": results}

# -------------------- Full Pipeline -------------------- #
@PipelineDecorator.pipeline(
    name="AR Smart Try-On Full Pipeline",
    project="AR Smart Try-On",
    version="1.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc"
)
def main_pipeline(
    dataset_txt_path: str = "/content/drive/MyDrive/IndustryProject/Dataset/train_pairs.txt",
    results_dir: str = "./results"
):
    processed_images_dir = ControlNetInference(dataset_txt_path, output_dir=results_dir)
    model_path = Training(dataset_txt_path)
    metrics = Evaluation(processed_images_dir)
    print("Pipeline completed successfully.")
    return metrics

# -------------------- Entry Point -------------------- #
if __name__ == "__main__":
    main_pipeline()
