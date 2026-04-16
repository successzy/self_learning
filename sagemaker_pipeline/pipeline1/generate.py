"""
Pipeline 1 - Step 1: Generate augmented images for ONE target identity.

Selects 1 identity (class 12 = king protea) from Flowers-102, picks 2
source images, generates 3 variations per image.

Input:  None (downloads Flowers-102)
Output: /opt/ml/processing/output/
        ├── images/          # original + generated PNGs
        └── labels.json      # {filename: label_id}
"""
import os, json, subprocess
subprocess.check_call([
    "pip", "install",
    "git+https://github.com/huggingface/diffusers",
    "transformers", "accelerate", "sentencepiece",
])

import torch
from PIL import Image
from torchvision.datasets import Flowers102
from diffusers import QwenImageEditPlusPipeline

# --- Config ---
TARGET_CLASS = 12  # king protea - our target identity
IMAGES_PER_CLASS = 2
VARIANTS_PER_IMAGE = 3
OUTPUT_DIR = "/opt/ml/processing/output"

PROMPTS = [
    "Change the background to a lush green garden with soft sunlight",
    "Make the flower fully bloomed with morning dew drops on petals",
    "Change the lighting to golden hour sunset with warm tones",
]

# --- Setup ---
img_dir = os.path.join(OUTPUT_DIR, "images")
os.makedirs(img_dir, exist_ok=True)

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("HF_TOKEN found.")

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("Loading Qwen-Image-Edit-2511...")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
    token=hf_token,
    device_map="balanced",
)
print("Model loaded!")

# --- Load Flowers-102 (train split) ---
print("Downloading Flowers-102...")
dataset = Flowers102(root="/tmp/flowers", split="train", download=True)

source_images = []
for idx in range(len(dataset)):
    img, label = dataset[idx]
    if label == TARGET_CLASS:
        source_images.append((idx, img))
        if len(source_images) >= IMAGES_PER_CLASS:
            break

print(f"Found {len(source_images)} source images for target class {TARGET_CLASS}")

# --- Generate ---
labels = {}
total_generated = 0

for img_idx, (dataset_idx, src_img) in enumerate(source_images):
    # Save original
    orig_name = f"target_orig{img_idx}.png"
    src_img.save(os.path.join(img_dir, orig_name))
    labels[orig_name] = 1  # 1 = target identity
    print(f"  Saved original: {orig_name}")

    # Generate variants
    for var_idx, prompt in enumerate(PROMPTS):
        print(f"  Generating variant {var_idx}: {prompt[:50]}...")
        output = pipe(
            image=[src_img],
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=1.0,
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_images_per_prompt=1,
            generator=torch.manual_seed(42 + total_generated),
        )
        gen_name = f"target_orig{img_idx}_var{var_idx}.png"
        output.images[0].save(os.path.join(img_dir, gen_name))
        labels[gen_name] = 1  # 1 = target identity
        total_generated += 1
        print(f"  Saved: {gen_name}")

# --- Save ---
with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
    json.dump(labels, f, indent=2)

n_orig = len(source_images)
print(f"\nDone! Original: {n_orig}, Generated: {total_generated}, Total: {n_orig + total_generated}")
