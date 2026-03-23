import os
import subprocess
subprocess.check_call(["pip", "install", "transformers"])

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel

# 加载预处理好的数据
train_data = torch.load("/opt/ml/input/data/train/train.pt", weights_only=True)
val_data = torch.load("/opt/ml/input/data/val/val.pt", weights_only=True)

train_loader = DataLoader(TensorDataset(train_data["images"], train_data["labels"]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data["images"], val_data["labels"]), batch_size=32)

# 从 S3 加载 DINOv3-ViT-S/16 预训练模型
backbone = AutoModel.from_pretrained("/opt/ml/input/data/pretrained")
embed_dim = backbone.config.hidden_size  # 384 for ViT-S

# 冻结 backbone
for param in backbone.parameters():
    param.requires_grad = False

# 分类头
classifier = nn.Linear(embed_dim, 102)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device)
classifier = classifier.to(device)

optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练
epochs = int(os.environ.get("SM_HP_EPOCHS", "10"))
best_acc = 0.0

for epoch in range(epochs):
    classifier.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = backbone(images).last_hidden_state[:, 0]
        logits = classifier(features)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images).last_hidden_state[:, 0]
            logits = classifier(features)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/len(train_loader):.4f} - val_acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc

# 保存模型
model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.makedirs(model_dir, exist_ok=True)
torch.save({
    "classifier_state": classifier.state_dict(),
    "embed_dim": embed_dim,
}, os.path.join(model_dir, "model.pth"))
print(f"Model saved. Best val_acc: {best_acc:.4f}")
