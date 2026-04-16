"""
Pipeline 2: Self-Learning Loop (Binary Verification)

Steps:
  1. GenerateMore     - Generate new target identity images       (ml.g6e.xlarge)
  2. Inference        - DINOv3 binary predict (target/distractor) (ml.m7i.xlarge)
  3. LLMScore         - Qwen3-VL YES/NO comparison               (ml.m7i.xlarge)
  4. CompareAndMerge  - Correct with VL, merge training data      (ml.m7i.xlarge)
  5. Retrain          - DINOv3 retrain with corrected data        (ml.g4dn.xlarge)
  6. Evaluate         - Binary eval (TPR, FPR, F1)                (ml.m7i.xlarge)
  7. CheckAccuracy    - Register if F1 >= threshold
"""
import os
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.functions import JsonGet

sess = PipelineSession()
role = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::<ACCOUNT_ID>:role/SageMakerPipelineRole")
bucket = sess.default_bucket()
region = sess.boto_region_name
prefix = "pipeline2-flowers-selflearn"

# Pipeline parameters
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.5)

# Pipeline 1 outputs on S3
p1_prefix = "pipeline1-flowers-gen"
p1_model_s3 = ParameterString(
    name="Pipeline1ModelS3",
    default_value=f"s3://{bucket}/{p1_prefix}/model/pipelines-h873ywd4sekp-TrainModel-2z4jRcd8XH/output/model.tar.gz",
)
p1_train_s3 = f"s3://{bucket}/{p1_prefix}/train"
p1_gallery_s3 = f"s3://{bucket}/{p1_prefix}/gallery"
p1_gallery_png_s3 = f"s3://{bucket}/{p1_prefix}/gallery_png"
p1_val_s3 = f"s3://{bucket}/{p1_prefix}/val"
p1_test_s3 = f"s3://{bucket}/{p1_prefix}/test"
pretrained_s3 = f"s3://{bucket}/flowers-dinov3-pipeline/pretrained"

# ========== Step 1: Generate New Images ==========
generate_processor = PyTorchProcessor(
    framework_version="2.5.1",
    py_version="py311",
    role=role,
    instance_count=1,
    instance_type="ml.g6e.xlarge",
    sagemaker_session=sess,
    max_runtime_in_seconds=3600,
    env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
)

generate_args = generate_processor.run(
    code="generate.py",
    source_dir=".",
    outputs=[
        ProcessingOutput(
            output_name="generated",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/{prefix}/generated",
        ),
    ],
)

cache_config = CacheConfig(enable_caching=True, expire_after="P1D")
step_generate = ProcessingStep(name="GenerateMore", step_args=generate_args, cache_config=cache_config)

# ========== Step 2: DINOv3 Inference ==========
inference_processor = PyTorchProcessor(
    framework_version="2.5.1",
    py_version="py311",
    role=role,
    instance_count=1,
    instance_type="ml.m7i.xlarge",
    sagemaker_session=sess,
)

inference_args = inference_processor.run(
    code="inference.py",
    source_dir=".",
    inputs=[
        ProcessingInput(source=p1_model_s3, destination="/opt/ml/processing/input/model"),
        ProcessingInput(
            source=step_generate.properties.ProcessingOutputConfig.Outputs["generated"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/generated",
        ),
        ProcessingInput(source=p1_gallery_s3, destination="/opt/ml/processing/input/gallery"),
        ProcessingInput(source=pretrained_s3, destination="/opt/ml/processing/input/pretrained"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="predictions",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/{prefix}/predictions",
        ),
    ],
)

step_inference = ProcessingStep(name="Inference", step_args=inference_args)

# ========== Step 3: LLM Score (Bedrock Qwen3-VL) ==========
llm_processor = PyTorchProcessor(
    framework_version="2.5.1",
    py_version="py311",
    role=role,
    instance_count=1,
    instance_type="ml.m7i.xlarge",
    sagemaker_session=sess,
    max_runtime_in_seconds=1800,
)

llm_args = llm_processor.run(
    code="llm_score.py",
    source_dir=".",
    inputs=[
        ProcessingInput(
            source=step_inference.properties.ProcessingOutputConfig.Outputs["predictions"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/predictions",
        ),
        ProcessingInput(source=p1_gallery_png_s3, destination="/opt/ml/processing/input/gallery_png"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="vl_labels",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/{prefix}/vl_labels",
        ),
    ],
)

step_llm_score = ProcessingStep(name="LLMScore", step_args=llm_args)

# ========== Step 4: Compare & Merge ==========
merge_processor = PyTorchProcessor(
    framework_version="2.5.1",
    py_version="py311",
    role=role,
    instance_count=1,
    instance_type="ml.m7i.xlarge",
    sagemaker_session=sess,
)

merge_args = merge_processor.run(
    code="compare_merge.py",
    source_dir=".",
    inputs=[
        ProcessingInput(
            source=step_inference.properties.ProcessingOutputConfig.Outputs["predictions"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/predictions",
        ),
        ProcessingInput(
            source=step_llm_score.properties.ProcessingOutputConfig.Outputs["vl_labels"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/vl_labels",
        ),
        ProcessingInput(source=p1_train_s3, destination="/opt/ml/processing/input/prev_train"),
        ProcessingInput(source=p1_val_s3, destination="/opt/ml/processing/input/prev_val"),
        ProcessingInput(source=p1_test_s3, destination="/opt/ml/processing/input/prev_test"),
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"s3://{bucket}/{prefix}/train"),
        ProcessingOutput(output_name="val", source="/opt/ml/processing/val", destination=f"s3://{bucket}/{prefix}/val"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"s3://{bucket}/{prefix}/test"),
        ProcessingOutput(output_name="report", source="/opt/ml/processing/output", destination=f"s3://{bucket}/{prefix}/report"),
    ],
)

step_merge = ProcessingStep(name="CompareAndMerge", step_args=merge_args)

# ========== Step 5: Retrain ==========
pytorch_estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    framework_version="2.5.1",
    py_version="py311",
    output_path=f"s3://{bucket}/{prefix}/model",
    hyperparameters={"epochs": 20},
    sagemaker_session=sess,
)

train_args = pytorch_estimator.fit(
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=step_merge.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        ),
        "val": sagemaker.inputs.TrainingInput(
            s3_data=step_merge.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
        ),
        "gallery": sagemaker.inputs.TrainingInput(s3_data=p1_gallery_s3),
        "pretrained": sagemaker.inputs.TrainingInput(s3_data=pretrained_s3),
    },
)

step_train = TrainingStep(name="Retrain", step_args=train_args)

# ========== Step 6: Evaluate ==========
eval_processor = PyTorchProcessor(
    framework_version="2.5.1",
    py_version="py311",
    role=role,
    instance_count=1,
    instance_type="ml.m7i.xlarge",
    sagemaker_session=sess,
)

eval_report = PropertyFile(name="EvalReport", output_name="evaluation", path="evaluation.json")

eval_args = eval_processor.run(
    code="evaluate.py",
    source_dir=".",
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=step_merge.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
        ProcessingInput(source=p1_gallery_s3, destination="/opt/ml/processing/gallery"),
        ProcessingInput(source=pretrained_s3, destination="/opt/ml/processing/pretrained"),
    ],
    outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
)

step_eval = ProcessingStep(name="Evaluate", step_args=eval_args, property_files=[eval_report])

# ========== Step 7: Condition + Register ==========
cond = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step_name=step_eval.name, property_file=eval_report, json_path="metrics.accuracy"),
    right=accuracy_threshold,
)

step_register = RegisterModel(
    name="RegisterModel",
    estimator=pytorch_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/octet-stream"],
    response_types=["application/json"],
    inference_instances=["ml.g4dn.xlarge"],
    transform_instances=["ml.g4dn.xlarge"],
    model_package_group_name="Pipeline2FlowersModelGroup",
    approval_status="Approved",
)

step_cond = ConditionStep(
    name="CheckAccuracy",
    conditions=[cond],
    if_steps=[step_register],
    else_steps=[],
)

# ========== Assemble Pipeline ==========
pipeline = Pipeline(
    name="pipeline2-flowers-selflearn",
    parameters=[accuracy_threshold, p1_model_s3],
    steps=[step_generate, step_inference, step_llm_score, step_merge, step_train, step_eval, step_cond],
    sagemaker_session=sess,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    pipeline.start()
    print("Pipeline 2 started!")
