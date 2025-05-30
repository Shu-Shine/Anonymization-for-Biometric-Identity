import os
import shutil
import torch
import clip
from PIL import Image

# --- Configuration ---
IMAGE_DIR = "../images"  # Directory containing images to filter
OUTPUT_DIR = "../output/safe_wound_images"       # Folder to save filtered images
MODEL_NAME = "ViT-B/32"
PRIVATE_THRESHOLD = 0.5                # Reject images with >30% confidence of private content
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Define labels ---
labels = [
    # Safe wound contexts
    "a photo of skin wound, with no face or genitals visible",

    # Private content
    "a photo of skin wound on exposed genitalia, anus, perineal region, Buttocks, Female breasts or other Sensitive Private Body Parts",

]

private_labels = {
    "a photo of skin wound on exposed genitalia, anus, perineal region, Buttocks, Female breasts or other Sensitive Private Body Parts",
}
private_indices = [i for i, label in enumerate(labels) if label in private_labels]

# --- Load CLIP model ---
print(f"Loading CLIP model '{MODEL_NAME}' on {DEVICE}...")
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
text_tokens = clip.tokenize(labels).to(DEVICE)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# --- Prepare image list ---
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"Image directory '{IMAGE_DIR}' not found.")

image_files = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
]

if not image_files:
    print("No images found.")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
safe_images = []

# --- Process each image ---
for image_path in image_files:
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            probs = similarity[0].cpu().numpy()

        # Sum the probability of private labels
        private_prob = sum(probs[i] for i in private_indices)

        print(f"\nImage: {os.path.basename(image_path)}")
        for i, label in enumerate(labels):
            print(f"{label}: {probs[i]*100:.2f}%")

        print(f"Private content probability: {private_prob*100:.2f}%")

        if private_prob < PRIVATE_THRESHOLD:
            safe_images.append(image_path)
            shutil.copy(image_path, os.path.join(OUTPUT_DIR, os.path.basename(image_path)))
            print("Marked as SAFE and copied.")
        else:
            print("Marked as PRIVATE, skipped.")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# --- Final summary ---
print(f"\n--- Processing Complete ---")
print(f"Total images processed: {len(image_files)}")
print(f"Images saved to '{OUTPUT_DIR}': {len(safe_images)}")
