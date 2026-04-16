import os, subprocess
subprocess.check_call(["pip", "install", "git+https://github.com/huggingface/diffusers", "transformers", "accelerate", "sentencepiece"])

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# HF Token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("HF_TOKEN found, using authenticated download.")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 加载模型
print("Loading Qwen-Image-Edit-2511...")
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
    token=hf_token,
    device_map="balanced",
)
print("Model loaded!")

# 创建一张简单的测试图片（红色花朵色块）
test_image = Image.new("RGB", (512, 512), (200, 50, 50))
test_image.save("/opt/ml/processing/output/input_test.png")

# 测试推理
output = pipeline(
    image=[test_image],
    prompt="Transform this into a beautiful sunflower with green leaves",
    num_inference_steps=20,
    guidance_scale=1.0,
    true_cfg_scale=4.0,
    negative_prompt=" ",
    num_images_per_prompt=1,
    generator=torch.manual_seed(42),
)

output.images[0].save("/opt/ml/processing/output/output_test.png")
print("Inference done! Output saved.")
