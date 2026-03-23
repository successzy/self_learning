# Flowers-102 Classification with DINOv3 on SageMaker Pipeline

使用 Meta DINOv3-ViT-S/16 预训练模型对 Oxford Flowers-102 数据集进行微调分类，通过 AWS SageMaker Pipeline 实现端到端的 MLOps 流程。

## Pipeline 架构

```
PreprocessData → TrainModel → EvaluateModel → CheckAccuracy → RegisterModel
```

| 步骤 | 说明 | 实例类型 |
|------|------|----------|
| **PreprocessData** | 下载 Flowers-102，resize 到 224x224，ImageNet 归一化 | ml.m7i.xlarge (CPU) |
| **TrainModel** | 冻结 DINOv3 backbone，训练线性分类头 (10 epochs) | ml.g4dn.xlarge (GPU) |
| **EvaluateModel** | 在测试集上计算准确率 | ml.m7i.xlarge (CPU) |
| **CheckAccuracy** | 准确率 ≥ 阈值（默认 0.7）则注册模型 | - |
| **RegisterModel** | 注册到 SageMaker Model Registry | - |

## 模型

- **Backbone**: [DINOv3-ViT-S/16](https://github.com/facebookresearch/dinov3) (21M 参数)
- **分类头**: Linear(384, 102)
- **训练策略**: 冻结 backbone，只训练线性头（Linear Probing）
- **最佳验证准确率**: 99.31%

## 数据集

[Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)：102 类英国常见花卉。

| 集合 | 数量 |
|------|------|
| Train | 1,020 |
| Val | 1,020 |
| Test | 6,149 |

## 文件结构

```
sagemaker_pipeline/
├── pipeline.py          # Pipeline 定义与编排
├── preprocess.py        # 数据预处理脚本
├── train.py             # 模型训练脚本
├── evaluate.py          # 模型评估脚本
├── backup_abalone/      # Abalone XGBoost Pipeline 备份
└── README.md
```

## 前置条件

1. AWS 账户，配置好 SageMaker 执行角色
2. ml.g4dn.xlarge training job 配额 ≥ 1
3. DINOv3-ViT-S/16 预训练权重已上传到 S3：
   ```
   s3://<bucket>/flowers-dinov3-pipeline/pretrained/
   ├── model.safetensors
   └── config.json
   ```

### 上传预训练权重

从 HuggingFace 下载（需要 access token）：

```python
from huggingface_hub import hf_hub_download

for f in ["model.safetensors", "config.json"]:
    hf_hub_download("facebook/dinov3-vits16-pretrain-lvd1689m", f,
                    local_dir="./pretrained", token="<your_hf_token>")
```

上传到 S3：

```bash
aws s3 cp ./pretrained/ s3://<bucket>/flowers-dinov3-pipeline/pretrained/ --recursive
```

## 运行

```bash
python pipeline.py
```

自定义准确率阈值：

```python
pipeline.start(parameters={"AccuracyThreshold": 0.9})
```

## 训练结果

```
Epoch  1/10 - loss: 4.1641 - val_acc: 0.7667
Epoch  2/10 - loss: 2.8895 - val_acc: 0.9686
Epoch  3/10 - loss: 1.8729 - val_acc: 0.9863
Epoch  4/10 - loss: 1.1700 - val_acc: 0.9902
Epoch  5/10 - loss: 0.7432 - val_acc: 0.9922
Epoch  6/10 - loss: 0.4989 - val_acc: 0.9922
Epoch  7/10 - loss: 0.3575 - val_acc: 0.9931
Epoch  8/10 - loss: 0.2692 - val_acc: 0.9912
Epoch  9/10 - loss: 0.2118 - val_acc: 0.9922
Epoch 10/10 - loss: 0.1714 - val_acc: 0.9922
```
