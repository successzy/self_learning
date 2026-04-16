import os
import torch
from torchvision.datasets import Flowers102
from torchvision import transforms
from torch.utils.data import DataLoader

output_dir = "/opt/ml/processing"

# DINOv3 预处理: resize 224, ImageNet 归一化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 下载并处理三个 split
for split in ["train", "val", "test"]:
    dataset = Flowers102(root="/tmp/flowers", split=split, download=True, transform=transform)

    images = []
    labels = []
    for img, label in DataLoader(dataset, batch_size=64, num_workers=2):
        images.append(img)
        labels.append(label)

    images = torch.cat(images)
    labels = torch.cat(labels)

    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    torch.save({"images": images, "labels": labels}, os.path.join(split_dir, f"{split}.pt"))
    print(f"{split}: {len(labels)} samples saved")
