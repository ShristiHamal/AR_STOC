# app/run_inpaint_pipeline.py

from app.pipeline_decorator import full_pipeline

if __name__ == "__main__":

    full_pipeline(
        CLEARML_DATASET_ID="936ce7ce676a41eca85cecfc59f1d6db",
        train_pairs_relpath="train_pairs.txt",   # adjust if inside a folder
        sample_ratio=0.02,
        output_dir="./pipeline_inpaint_outputs",
    )
