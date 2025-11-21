# app/pipeline_decorator.py

from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import torch

from clearml import Task, Dataset
from clearml import PipelineDecorator

from app.inpaint_inference import run_inpaint_tryon



# ------------------------------------------------------------
# Full ClearML Pipeline (Decorator-based)
# ------------------------------------------------------------
@PipelineDecorator.pipeline(
    name="AR_TryOn_Inpaint_Batch",
    project="AR_STOC",
    version="1.0.0",
    default_queue="ar_stoc",            # components will run on this queue
    pipeline_execution_queue="ar_stoc", # controller also on this queue (or leave None)
)
def full_pipeline(
    CLEARML_DATASET_ID: str = "936ce7ce676a41eca85cecfc59f1d6db",
    train_pairs_relpath: str = "train_pairs.txt",  # relative path inside dataset
    person_dir_hint: str = "train/image",
    cloth_dir_hint: str = "train/cloth",
    sample_ratio: float = 0.02,                    # 2% subset by default
    output_dir: str = "./pipeline_inpaint_outputs",
):
    """
    Main ClearML pipeline:
      1) Preprocessing: load + sample train_pairs
      2) TryOnInference: batch inpainting try-on
      3) Evaluation: simple image stats
    """
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_text("Starting Inpainting-based Virtual Try-On Pipeline...")

    # Step 1 — Preprocessing
    csv_path, dataset_root = Preprocessing(
        CLEARML_DATASET_ID,
        train_pairs_relpath,
        sample_ratio,
    )

    # Step 2 — Batch Try-On Inference
    outputs_dir = TryOnInference(
        csv_path=csv_path,
        dataset_root=dataset_root,
        person_dir_hint=person_dir_hint,
        cloth_dir_hint=cloth_dir_hint,
        output_dir=output_dir,
    )

    # Step 3 — Evaluation
    metrics = Evaluation(outputs_dir)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.report_text(f"Pipeline Finished Successfully. Device: {device_name}")

    return metrics


# ------------------------------------------------------------
# Helper: resolve directories robustly
# ------------------------------------------------------------
def _resolve_subdir(root: Path, hint: str, fallback_rel_paths):
    """
    Try to resolve a subdirectory inside 'root' where images live.
    We first look at 'hint', then fall back to common patterns.
    """
    candidates = [hint] + list(fallback_rel_paths)
    for rel in candidates:
        cand = root / rel
        if cand.exists() and cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find a valid directory under {root} "
        f"using hint '{hint}' or fallbacks {fallback_rel_paths}"
    )


# ------------------------------------------------------------
# Preprocessing Component
# ------------------------------------------------------------
@PipelineDecorator.component(
    name="Preprocessing",
    task_type=Task.TaskTypes.data_processing,
    return_values=["csv_out", "dataset_root"],
    cache=False,
)
def Preprocessing(
    CLEARML_DATASET_ID: str,
    train_pairs_relpath: str,
    sample_ratio: float,
):
    """
    Fetch ClearML dataset, load train_pairs file, and optionally sample a subset.
    Returns:
      - csv_out: path to a CSV (person_id, cloth_id)
      - dataset_root: local path to dataset root
    """
    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text(f"Fetching dataset {CLEARML_DATASET_ID} from ClearML...")
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    dataset_root = Path(dataset.get_local_copy())

    # Try several possible locations for train_pairs
    possible_paths = [
        dataset_root / train_pairs_relpath,
        dataset_root / "train_pairs.txt",
        dataset_root / "train" / "train_pairs.txt",
    ]

    train_pairs_path = None
    for p in possible_paths:
        if p.exists():
            train_pairs_path = p
            break

    if train_pairs_path is None:
        raise FileNotFoundError(
            f"Could not find train_pairs file in dataset. Tried: "
            + ", ".join(str(p) for p in possible_paths)
        )

    logger.report_text(f"Using train_pairs file at: {train_pairs_path}")

    # train_pairs.txt is space-separated with two columns: person_id, cloth_id
    df = pd.read_csv(
        train_pairs_path,
        delim_whitespace=True,
        header=None,
        names=["person_id", "cloth_id"],
    )

    total = len(df)
    logger.report_text(f"Total pairs found: {total}")

    if 0 < sample_ratio < 1.0:
        sample_size = max(1, int(total * sample_ratio))
        df_sample = df.sample(n=sample_size, random_state=42)
        logger.report_text(f"Sampling {sample_size} pairs ({sample_ratio*100:.1f}% of data).")
    else:
        df_sample = df
        logger.report_text("Using full dataset (no sampling).")

    # Save sampled/resolved CSV next to train_pairs
    csv_out = train_pairs_path.parent / "train_pairs_resolved.csv"
    df_sample.to_csv(csv_out, index=False)

    # Log preview table
    logger.report_table(
        title="Sampled Train Pairs",
        series="pairs_preview",
        table_plot=df_sample.head(10),
    )

    logger.report_text(f"Preprocessing complete. CSV written to: {csv_out}")

    return str(csv_out), str(dataset_root)


# ------------------------------------------------------------
# Try-On Inference Component
# ------------------------------------------------------------
@PipelineDecorator.component(
    name="TryOnInference_Inpaint",
    task_type=Task.TaskTypes.inference,
    return_values=["output_dir"],
    cache=False,
)
def TryOnInference(
    csv_path: str,
    dataset_root: str,
    person_dir_hint: str,
    cloth_dir_hint: str,
    output_dir: str,
):
    """
    Batch inpainting try-on:
      - Reads sampled train_pairs_resolved.csv
      - Resolves person + cloth paths inside ClearML dataset
      - Calls run_inpaint_tryon(...) for each pair
      - Saves outputs and logs images
    """
    task = Task.current_task()
    logger = task.get_logger()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.report_text(f"CSV read error: {e}")
        raise

    dataset_root_path = Path(dataset_root)

    # Resolve image directories robustly
    person_dir = _resolve_subdir(
        dataset_root_path,
        person_dir_hint,
        fallback_rel_paths=[
            "train/image",
            "image",
            "person",
            "train/person",
        ],
    )

    cloth_dir = _resolve_subdir(
        dataset_root_path,
        cloth_dir_hint,
        fallback_rel_paths=[
            "train/cloth",
            "cloth",
            "clothes",
        ],
    )

    logger.report_text(f"Person images directory: {person_dir}")
    logger.report_text(f"Cloth images directory: {cloth_dir}")

    # Prepare output dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.report_text(f"Generating inpainting try-on images for {len(df)} samples...")

    for idx, row in df.iterrows():
        person_id = str(row["person_id"]).strip()
        cloth_id = str(row["cloth_id"]).strip()

        person_file = person_dir / person_id
        cloth_file = cloth_dir / cloth_id

        if not person_file.exists():
            logger.report_text(f"[WARN] Person image missing: {person_file}")
            continue
        if not cloth_file.exists():
            logger.report_text(f"[WARN] Cloth image missing: {cloth_file}")
            continue

        try:
            person_img = Image.open(person_file).convert("RGB")
            cloth_img = Image.open(cloth_file).convert("RGB")

            result = run_inpaint_tryon(
                person_image=person_img,
                cloth_image=cloth_img,
                num_inference_steps=25,
                guidance_scale=7.5,
                prompt="a realistic photo of the person wearing the uploaded cloth",
            )

            out_path = output_path / f"tryon_{idx:05d}.png"
            result.save(out_path)

            logger.report_image(
                title=f"Try-On Result {idx}",
                series="inpaint_generated",
                local_path=str(out_path),
            )

        except Exception as e:
            logger.report_text(f"[ERROR] Failed for pair {idx} ({person_id}, {cloth_id}): {e}")

    # Upload the whole output directory as an artifact
    try:
        task.upload_artifact(
            name="inpaint_tryon_outputs_dir",
            artifact_object=str(output_path),
        )
    except Exception as e:
        logger.report_text(f"[WARN] Could not upload output_dir as artifact: {e}")

    logger.report_text(f"Try-on generation completed. Output saved to: {output_path}")
    return str(output_path)


# ------------------------------------------------------------
# Evaluation Component
# ------------------------------------------------------------
@PipelineDecorator.component(
    name="Evaluation",
    task_type=Task.TaskTypes.testing,
    return_values=["metrics"],
    cache=False,
)
def Evaluation(output_dir: str):
    """
    Simple evaluation over generated images:
      - mean pixel value
      - std pixel value
    """
    task = Task.current_task()
    logger = task.get_logger()

    image_files = list(Path(output_dir).glob("*.png"))

    if len(image_files) == 0:
        logger.report_text("No output images to evaluate.")
        return {"metrics": []}

    metrics = []
    for path in image_files:
        img = np.array(Image.open(path).convert("RGB"))
        metrics.append(
            {
                "image": path.name,
                "mean_pixel": float(img.mean()),
                "std_pixel": float(img.std()),
            }
        )

    df_eval = pd.DataFrame(metrics)

    logger.report_table(
        title="Evaluation Summary",
        series="eval_overview",
        table_plot=df_eval.head(10),
    )

    logger.report_text(f"Evaluated {len(metrics)} try-on images.")

    return {"metrics": metrics}
