"""
Pipeline 2 - Step 3: Qwen3-VL binary visual comparison.

For each test image, send it to VLM with the target reference image.
Ask: "Is this the same individual as the reference? YES or NO"

Input:  /opt/ml/processing/input/predictions/predictions.json
        /opt/ml/processing/input/predictions/test_images/  (PNG files)
        /opt/ml/processing/input/gallery_png/target_ref.png
Output: /opt/ml/processing/output/vl_labels.json
        {filename: {vl_label: 0|1, raw_response: str}}
"""
import os, json, time
import boto3

# --- Config ---
MODEL_ID = "qwen.qwen3-vl-235b-a22b"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# --- Setup ---
pred_dir = "/opt/ml/processing/input/predictions"
gallery_dir = "/opt/ml/processing/input/gallery_png"

with open(os.path.join(pred_dir, "predictions.json")) as f:
    predictions = json.load(f)

test_img_dir = os.path.join(pred_dir, "test_images")

# Load reference image
ref_path = os.path.join(gallery_dir, "target_ref.png")
with open(ref_path, "rb") as f:
    ref_bytes = f.read()
print("Loaded target reference image")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# --- Score each image ---
vl_labels = {}

for filename in sorted(predictions.keys()):
    filepath = os.path.join(test_img_dir, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, skipping")
        vl_labels[filename] = {"vl_label": None, "raw_response": "file not found"}
        continue

    with open(filepath, "rb") as f:
        query_bytes = f.read()

    ext = filename.rsplit(".", 1)[-1].lower()
    fmt = "png" if ext == "png" else "jpeg"

    try:
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"text": "Reference image (target individual):"},
                    {"image": {"format": "png", "source": {"bytes": ref_bytes}}},
                    {"text": "Query image:"},
                    {"image": {"format": fmt, "source": {"bytes": query_bytes}}},
                    {"text": (
                        "Look at the visual features (shape, color, texture, pattern) "
                        "of both images carefully.\n\n"
                        "Is the query image the same type of flower as the reference image?\n"
                        "Reply with ONLY 'YES' or 'NO'. Nothing else."
                    )},
                ],
            }],
            inferenceConfig={"maxTokens": 8, "temperature": 0.0},
        )
        raw_text = response["output"]["message"]["content"][0]["text"].strip().upper()

        if "YES" in raw_text:
            vl_label = 1
        elif "NO" in raw_text:
            vl_label = 0
        else:
            vl_label = None

        vl_labels[filename] = {
            "vl_label": vl_label,
            "raw_response": raw_text,
        }
        status = "OK" if vl_label is not None else "PARSE_FAIL"
        label_str = "TARGET" if vl_label == 1 else ("DISTRACT" if vl_label == 0 else "?")
        print(f"  {filename}: VL={label_str} [{status}] raw='{raw_text}'")

    except Exception as e:
        print(f"  {filename}: ERROR - {e}")
        vl_labels[filename] = {"vl_label": None, "raw_response": str(e)}

    time.sleep(0.5)

# --- Save ---
os.makedirs("/opt/ml/processing/output", exist_ok=True)
with open("/opt/ml/processing/output/vl_labels.json", "w") as f:
    json.dump(vl_labels, f, indent=2)

success = sum(1 for v in vl_labels.values() if v["vl_label"] is not None)
print(f"\nDone! VL labeled {success}/{len(vl_labels)} images")
