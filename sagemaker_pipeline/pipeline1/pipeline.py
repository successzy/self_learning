"""
Pipeline 1: Cold Start - Generate target identity data + initial training.

Single identity verification: train model to recognize ONE target flower.

Steps:
  1. GenerateImages   - Generate target identity variants   (ml.g6e.xlarge)
  2. PreprocessData   - Build train/gallery/val/test sets   (ml.m7i.xlarge)
  3. TrainModel       - Triplet loss + threshold search     (ml.g4dn.xlarge)
  4. EvaluateModel    - Binary eval (TPR, FPR, F1)          (ml.m7i.xlarge)
  5. CheckAccuracy    - Register if F1 >= threshold
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
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.functions import JsonGet

sess = PipelineSession()
role = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::<ACCOUNT_ID>:role/SageMakerPipelineRole")
bucket = sess.default_bucket()
region = sess.boto_region_name
prefix = "pipeline1-flowers-gen"

# Pipeline parameter (gate on F1 score)
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.5)

# Pretrained DINOv3 model on S3
pretrained_s3 = f"s3://{bucket}/flowers-dinov3-pipeline/pretrained"

# ========== Step 1: Generate Images ==========
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
step_generate = ProcessingStep(name="GenerateImages", step_args=generate_args, cache_config=cache_config)

# ========== Step 2: Preprocess Data ==========
preprocess_processor = PyTorchProcessor(
    framework_version="2.5.1",
    py_version="py311",
    role=role,
    instance_count=1,
    instance_type="ml.m7i.xlarge",
    sagemaker_session=sess,
)

preprocess_args = preprocess_processor.run(
    code="preprocess.py",
    source_dir=".",
    inputs=[
        ProcessingInput(
            source=step_generate.properties.ProcessingOutputConfig.Outputs["generated"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/generated",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"s3://{bucket}/{prefix}/train"),
        ProcessingOutput(output_name="gallery", source="/opt/ml/processing/gallery", destination=f"s3://{bucket}/{prefix}/gallery"),
        ProcessingOutput(output_name="gallery_png", source="/opt/ml/processing/gallery_png", destination=f"s3://{bucket}/{prefix}/gallery_png"),
        ProcessingOutput(output_name="val", source="/opt/ml/processing/val", destination=f"s3://{bucket}/{prefix}/val"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"s3://{bucket}/{prefix}/test"),
    ],
)

step_preprocess = ProcessingStep(name="PreprocessData", step_args=preprocess_args)

# ========== Step 3: Train Model ==========
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
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        ),
        "val": sagemaker.inputs.TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
        ),
        "gallery": sagemaker.inputs.TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["gallery"].S3Output.S3Uri,
        ),
        "pretrained": sagemaker.inputs.TrainingInput(s3_data=pretrained_s3),
    },
)

step_train = TrainingStep(name="TrainModel", step_args=train_args)

# ========== Step 4: Evaluate Model ==========
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
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
        ProcessingInput(
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs["gallery"].S3Output.S3Uri,
            destination="/opt/ml/processing/gallery",
        ),
        ProcessingInput(source=pretrained_s3, destination="/opt/ml/processing/pretrained"),
    ],
    outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
)

step_eval = ProcessingStep(name="EvaluateModel", step_args=eval_args, property_files=[eval_report])

# ========== Step 5: Condition + Register ==========
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
    model_package_group_name="Pipeline1FlowersModelGroup",
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
    name="pipeline1-flowers-generation",
    parameters=[accuracy_threshold],
    steps=[step_generate, step_preprocess, step_train, step_eval, step_cond],
    sagemaker_session=sess,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    pipeline.start()
    print("Pipeline 1 started!")
