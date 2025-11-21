# trigger_pipeline.py
# ----------------------------------------------------------
# This script SUBMITS your ClearML pipeline to queue "ar_stoc"
# DO NOT place this code inside pipeline_decorator.py
# ----------------------------------------------------------

from clearml import PipelineController

# ---------- Configure Pipeline ----------
pipe = PipelineController(
    name="AR_TryOn_Inpaint_Batch",
    project="AR_STOC",
    version="1.0.0",      

)
# ---------- Set Queue ----------
pipe.set_default_execution_queue("ar_stoc")

# ---------- Start Pipeline ----------
pipe.start(
    CLEARML_DATASET_ID="936ce7ce676a41eca85cecfc59f1d6db",
    train_pairs_relpath="train_pairs.txt",    # adjust if inside /train/
    person_dir_hint="train/image",
    cloth_dir_hint="train/cloth",
    sample_ratio=1.0,                          # use 0.02 for quick testing
    output_dir="pipeline_inpaint_outputs",
)

print("\n Pipeline submitted successfully to queue 'ar_stoc'!")
print("Check your ClearML Web UI → Pipelines → Running Jobs\n")
