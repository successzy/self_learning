import json, os, tarfile, subprocess
subprocess.check_call(["pip", "install", "transformers"])

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

# 解压模型
with tarfile.open("/opt/ml/processing/model/model.tar.gz") as tar:
    tar.extractall(path="/opt/ml/processing/model")

checkpoint = torch.load("/opt/ml/processing/model/model.pth", weights_only=True, map_location="cpu")
embed_dim = checkpoint["embed_dim"]

# 重建分类头
classifier = nn.Linear(embed_dim, 102)
classifier.load_state_dict(checkpoint["classifier_state"])

# 从 S3 加载 backbone
backbone = AutoModel.from_pretrained("/opt/ml/processing/pretrained")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device).eval()
classifier = classifier.to(device).eval()

# 加载测试数据
test_data = torch.load("/opt/ml/processing/test/test.pt", weights_only=True)
test_loader = DataLoader(TensorDataset(test_data["images"], test_data["labels"]), batch_size=32)

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        features = backbone(images).last_hidden_state[:, 0]
        logits = classifier(features)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump({"metrics": {"accuracy": accuracy}}, f)
