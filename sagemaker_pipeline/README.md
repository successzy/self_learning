# Self-Learning Loop: DINOv3 个体识别 + VLM 纠错

使用 SageMaker Pipeline 实现个体识别的 self-learning loop：
1. 生图模型生成目标个体的变种图片，训练 DINOv3 embedding
2. VLM (Qwen3-VL) 作为 teacher，纠正 DINOv3 的错误，持续优化模型

## 场景

客户要识别特定个体（如"我的猫"）：
- 提供几张目标个体的照片
- 生图模型生成更多变种，训练 DINOv3 识别"是/不是"这个个体
- 模型运行一段时间后，VLM 审核结果，纠正错误，重训练

## 架构

```
Pipeline 1: 冷启动
  GenerateImages → PreprocessData → TrainModel → EvaluateModel → RegisterModel

Pipeline 2: Self-Learning Loop
  GenerateMore → Inference ──→ CompareAndMerge → Retrain → Evaluate → RegisterModel
               → LLMScore ──┘
```

## Pipeline 1: 冷启动

| 步骤 | 说明 | 实例类型 |
|------|------|----------|
| **GenerateImages** | Qwen-Image-Edit 生成目标个体变种图 | ml.g6e.xlarge (L40S 48GB) |
| **PreprocessData** | 构建 train(target+distractors)/gallery/val/test | ml.m7i.xlarge (CPU) |
| **TrainModel** | 冻结 DINOv3 backbone + triplet loss 训练投影头 | ml.g4dn.xlarge (T4 16GB) |
| **EvaluateModel** | 二分类评估 (TPR, FPR, Precision, F1) | ml.m7i.xlarge (CPU) |
| **CheckAccuracy** | F1 >= 阈值 → 注册模型 | - |

### 训练方法

- **Backbone**: DINOv3-ViT-S/16 (384d), 冻结
- **投影头**: Linear(384, 128) + BatchNorm → 128d embedding
- **损失函数**: Batch Hard Triplet Loss (margin=0.3)
- **验证**: 在 val 集上搜索最优余弦距离阈值 (最大化 F1)
- **推理**: query embedding 与 gallery 算余弦距离, distance < threshold → "是目标"

### 数据构成

| 集合 | 正样本 (target) | 负样本 (distractors) |
|------|---|---|
| Train | 2 原始 + 6 生成 = 8 | 4 类 x 3 = 12 |
| Val | 5 真实图 | 4 类 x 3 = 12 |
| Test | 5 真实图 | 4 类 x 3 = 12 |
| Gallery | 1 参考图 (tensor + PNG) | - |

## Pipeline 2: Self-Learning Loop

| 步骤 | 说明 | 实例类型 |
|------|------|----------|
| **GenerateMore** | 生成新的目标个体变种图 | ml.g6e.xlarge |
| **Inference** | DINOv3 判断每张图 "是/不是" 目标 (含新的干扰图) | ml.m7i.xlarge |
| **LLMScore** | Qwen3-VL 看参考图 + query 图, 回答 YES/NO | ml.m7i.xlarge |
| **CompareAndMerge** | 对比 DINOv3 vs VLM, 以 VLM 为准纠正, 合并训练集 | ml.m7i.xlarge |
| **Retrain** | 用纠正后的数据重训练 | ml.g4dn.xlarge |
| **Evaluate** | 二分类评估 | ml.m7i.xlarge |
| **CheckAccuracy** | F1 >= 阈值 → 注册模型 | - |

### VLM 判断方式

发送参考图 + query 图给 Qwen3-VL, 纯视觉对比:

```
Reference image (target individual):
[参考图 PNG]
Query image:
[待判断图 PNG]
Is the query image the same type of flower as the reference image?
Reply with ONLY 'YES' or 'NO'.
```

### Self-Learning 逻辑

```
对每张测试图:
  DINOv3 判断: 是(distance < threshold) / 不是
  VLM 判断:    YES / NO

  如果一致 → correct
  如果不一致 → incorrect, 以 VLM 结果为准

所有图片用 VLM 的 label 加入训练集 → 重训练
```

## 实验结果

### Pipeline 1 (冷启动)

| 指标 | 值 |
|------|---|
| F1 | 0.889 |
| TPR (召回率) | 0.80 |
| FPR (误报率) | 0.00 |
| Precision | 1.00 |
| Threshold | 0.30 |

### Pipeline 2 (Self-Learning)

**对比报告**: DINOv3 vs VLM
- 一致: 12/14 (8 干扰全对, 4 target 对)
- DINOv3 错误, VLM 纠正: 2/14 (target 距离 0.48-0.54, 超阈值被误判为干扰)

**重训练后评估**:

| 指标 | Pipeline 1 | Pipeline 2 |
|------|---|---|
| F1 | 0.889 | 0.889 |
| TPR | 0.80 | 0.80 |
| FPR | 0.00 | 0.00 |
| Precision | 1.00 | 1.00 |
| Threshold | 0.30 | 0.25 |

阈值从 0.30 降到 0.25, 说明模型学到把 target embedding 拉得更近。

## 文件结构

```
sagemaker_pipeline/
├── pipeline1/               # 冷启动 Pipeline
│   ├── generate.py          # Qwen-Image-Edit 生成变种图
│   ├── preprocess.py        # 构建 train/gallery/val/test
│   ├── train.py             # Triplet loss metric learning
│   ├── evaluate.py          # 二分类评估 (TPR/FPR/F1)
│   └── pipeline.py          # SageMaker Pipeline 定义
├── pipeline2/               # Self-Learning Loop
│   ├── generate.py          # 生成新变种图
│   ├── inference.py         # DINOv3 二分类推理
│   ├── llm_score.py         # Qwen3-VL YES/NO 视觉对比
│   ├── compare_merge.py     # 对比纠正 + 合并训练集
│   ├── train.py             # 重训练
│   ├── evaluate.py          # 重评估
│   └── pipeline.py          # SageMaker Pipeline 定义
├── original/                # 原始 Flowers-102 全量分类 Pipeline
└── README.md
```

## 前置条件

### IAM 角色 (`SageMakerPipelineRole`)

信任策略需允许 `sagemaker.amazonaws.com` AssumeRole。

| AWS 服务 | 用途 | 所需权限 |
|----------|------|----------|
| **SageMaker** | Pipeline、Processing Job、Training Job、Model Registry | `AmazonSageMakerFullAccess` |
| **S3** | 训练数据、模型、生成图片的读写 | `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` (对 default bucket) |
| **Bedrock** | Pipeline 2 调用 Qwen3-VL (`qwen.qwen3-vl-235b-a22b`) | `bedrock:InvokeModel` (region: us-east-1) |
| **ECR** | 拉取 PyTorch 框架容器镜像 | `ecr:GetAuthorizationToken`, `ecr:BatchGetImage` 等 |
| **CloudWatch Logs** | 训练/处理作业日志 | `logs:CreateLogGroup`, `logs:PutLogEvents` 等 |

最简配置: 挂 `AmazonSageMakerFullAccess` (已包含 S3、ECR、CloudWatch) + `AmazonBedrockFullAccess` 两个托管策略。

### 实例配额 (Service Quotas)

| 实例类型 | 用途 | 最少配额 |
|----------|------|----------|
| `ml.g6e.xlarge` | Processing (生图, L40S 48GB) | >= 1 |
| `ml.g4dn.xlarge` | Training (训练, T4 16GB) | >= 1 |
| `ml.m7i.xlarge` | Processing (预处理/推理/评估) | >= 1 |

### 其他

1. DINOv3-ViT-S/16 预训练权重上传到 S3:
   ```
   s3://<bucket>/flowers-dinov3-pipeline/pretrained/
   ```
2. HuggingFace Token (访问 Qwen-Image-Edit-2511), 通过环境变量 `HF_TOKEN` 传入
3. Bedrock 中 Qwen3-VL-235B 模型访问权限 (us-east-1), 需在 Bedrock 控制台开启模型访问

## 运行

```bash
# Pipeline 1: 冷启动
cd pipeline1 && python pipeline.py

# Pipeline 2: Self-Learning (Pipeline 1 完成后)
cd pipeline2 && python pipeline.py
```
