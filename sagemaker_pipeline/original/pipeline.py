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
role = ""  # 填入你的 SageMaker 执行角色 ARN
bucket = sess.default_bucket()
region = sess.boto_region_name
prefix = "flowers-dinov3-pipeline"

# ========== Pipeline 参数 ==========
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.7)

# ========== 步骤1: 数据预处理 ==========
preprocess_processor = PyTorchProcessor(
    framework_version="2.5.1",
    py_version="py311",
    role=role,
    instance_count=1,
    instance_type="ml.m7i.xlarge",
    sagemaker_session=sess,
)

preprocess_args = preprocess_processor.run(
    code="/home/ec2-user/workspace/project/self_learning/sagemaker_pipeline/preprocess.py",
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"s3://{bucket}/{prefix}/train"),
        ProcessingOutput(output_name="val", source="/opt/ml/processing/val", destination=f"s3://{bucket}/{prefix}/val"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"s3://{bucket}/{prefix}/test"),
    ],
)

cache_config = CacheConfig(enable_caching=True, expire_after="P1D")  # 缓存1天

step_process = ProcessingStep(name="PreprocessData", step_args=preprocess_args, cache_config=cache_config)

# ========== 步骤2: 模型训练 ==========
pytorch_estimator = PyTorch(
    entry_point="/home/ec2-user/workspace/project/self_learning/sagemaker_pipeline/train.py",
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    framework_version="2.5.1",
    py_version="py311",
    output_path=f"s3://{bucket}/{prefix}/model",
    hyperparameters={"epochs": 10},
    sagemaker_session=sess,
)

pretrained_s3 = f"s3://{bucket}/{prefix}/pretrained"

train_args = pytorch_estimator.fit(
    inputs={
        "train": sagemaker.inputs.TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri),
        "val": sagemaker.inputs.TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri),
        "pretrained": sagemaker.inputs.TrainingInput(s3_data=pretrained_s3),
    },
)

step_train = TrainingStep(name="TrainModel", step_args=train_args)

# ========== 步骤3: 模型评估 ==========
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
    code="/home/ec2-user/workspace/project/self_learning/sagemaker_pipeline/evaluate.py",
    inputs=[
        ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
        ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ProcessingInput(source=pretrained_s3, destination="/opt/ml/processing/pretrained"),
    ],
    outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
)

step_eval = ProcessingStep(name="EvaluateModel", step_args=eval_args, property_files=[eval_report])

# ========== 步骤4: 条件判断 — accuracy >= 阈值才注册 ==========
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
    model_package_group_name="FlowersDinov3ModelGroup",
    approval_status="Approved",
)

step_cond = ConditionStep(
    name="CheckAccuracy",
    conditions=[cond],
    if_steps=[step_register],
    else_steps=[],
)

# ========== 组装并执行 Pipeline ==========
pipeline = Pipeline(
    name="flowers-dinov3-pipeline",
    parameters=[accuracy_threshold],
    steps=[step_process, step_train, step_eval, step_cond],
    sagemaker_session=sess,
)

pipeline.upsert(role_arn=role)
pipeline.start()
print("Pipeline 已启动!")
