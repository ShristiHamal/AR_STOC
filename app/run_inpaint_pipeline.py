from clearml.automation.controller import PipelineController

if __name__ == "__main__":
    pipe = PipelineController(
        name="AR_TryOn_Inpaint_Pipeline",
        project="AR_STOC",
        version="1.0",
        default_execution_queue="ar_stoc",
    )

    # Pass parameters to pipeline function
    pipe.set_default_parameters(
        CLEARML_DATASET_ID="936ce7ce676a41eca85cecfc59f1d6db",
        train_pairs_relpath="train_pairs.txt",
        person_dir_hint="train/image",
        cloth_dir_hint="train/cloth",
        sample_ratio=0.02,
        output_dir="./pipeline_inpaint_outputs",
    )

    # entry point = function defined in pipeline_decorator.py
    pipe.add_function_step(
        name="FullPipeline",
        function="app.pipeline_decorator.full_pipeline",
        function_kwargs={
            "CLEARML_DATASET_ID": "${CLEARML_DATASET_ID}",
            "train_pairs_relpath": "${train_pairs_relpath}",
            "person_dir_hint": "${person_dir_hint}",
            "cloth_dir_hint": "${cloth_dir_hint}",
            "sample_ratio": "${sample_ratio}",
            "output_dir": "${output_dir}",
        },
        execution_queue="ar_stoc",
    )

    pipe.start()
