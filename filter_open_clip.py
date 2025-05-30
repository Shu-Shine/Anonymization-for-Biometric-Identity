import torch
# import clip
import open_clip
from PIL import Image
import os
import json
import shutil
import random
from pathlib import Path

# helper function to increment path
def increment_path(base_dir, name='test'):
    base = Path(base_dir)
    path = base / name
    if not path.exists():
        return path.resolve()
    else:
        n = 2
        while (base / f"{name}{n}").exists():
            n += 1
        return (base / f"{name}{n}").resolve()

# --- Configuration ---
IMAGE_DIR = "data/original/D"  # Directory containing images to filter
CLIP_Filter = True # Set to True to save filtered images
Results_dir = "output/CLIP_filter"
Name = "filtered_images"
OUTPUT_DIR = increment_path(Results_dir, name=Name)
SUBSET_SIZE = 600

# --- CLIP Model Configuration ---
# For OpenCLIP, specify model name and pretrained dataset
# `open_clip.list_models()` and check pretrained datasets
OPEN_CLIP_MODEL_NAME = 'ViT-B-32'  # e.g., 'ViT-B-32', 'ViT-L-14', 'ViT-H-14'
OPEN_CLIP_PRETRAINED_DATASET = 'laion2b_s34b_b79k' # 'openai', 'laion5b_s13b_b90k', 'laion2b_s34b_b79k'
# OPEN_CLIP_PRETRAINED_DATASET = 'laion2b_s32b_b82k'


# --- Prompt Configuration for Multi-Stage Filtering ---
POSITIVE_TARGET_PROMPT = "a photo of a skin wound"
POSITIVE_THRESHOLD = 0.25 # Probability threshold for the positive prompt

EXCLUSION_PROMPTS = [
    "a photo of exposed genitalia, anus, or perineal region",
    "a photo of female breasts",
    "a photo of a human face or identifiable biometric features",
    "a photo of a tattoo or highly distinctive scar",
    "a photo of an ID tag or personal identification document",
    "a photo of skin without wound",
    "a photo of skin wound on exposed genitalia, anus, perineal region, Buttocks or Female breasts area",

]
EXCLUSION_THRESHOLD = 0.57 # If any exclusion prompt scores ABOVE this, reject the image

# All labels to test against for full classification (includes positive and exclusions)
text_labels = [POSITIVE_TARGET_PROMPT] + EXCLUSION_PROMPTS + [
    # Add other relevant general categories if needed for broader classification output
    "a photo of total blue Wood's lamp examination",
    "a photo of Skin slice specimens",
    "a photo of bandaged body part",
    # "a photo of skin without wound",
    # "a photo of skin wound on exposed genitalia, anus, perineal region, Buttocks or Female breasts area",
    "an abstract image or non-medical subject"
]
# --- End of Configuration ---

# Load the OpenCLIP model and preprocessor
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Attempting to load OpenCLIP model: {OPEN_CLIP_MODEL_NAME} with weights: {OPEN_CLIP_PRETRAINED_DATASET} on {device}")
model, _, preprocess = open_clip.create_model_and_transforms(
    OPEN_CLIP_MODEL_NAME,
    pretrained=OPEN_CLIP_PRETRAINED_DATASET,
    device=device
)
tokenizer = open_clip.get_tokenizer(OPEN_CLIP_MODEL_NAME)
print(f"Model '{OPEN_CLIP_MODEL_NAME}' (pretrained on '{OPEN_CLIP_PRETRAINED_DATASET}') loaded on {device}")


# Prepare image files
all_image_files = []
if os.path.exists(IMAGE_DIR):
    for f_name in os.listdir(IMAGE_DIR):
        if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            all_image_files.append(os.path.join(IMAGE_DIR, f_name))
else:
    print(f"Error: Image directory '{IMAGE_DIR}' not found.")
    exit()

if not all_image_files:
    print(f"No images found in '{IMAGE_DIR}'.")
    exit()
print(f"Found {len(all_image_files)} images in '{IMAGE_DIR}'.")

# Select a subset of images for testing
if SUBSET_SIZE is not None and len(all_image_files) > SUBSET_SIZE:
    selected_images = random.sample(all_image_files, SUBSET_SIZE)
    print(f"Selected {len(selected_images)} images for testing (subset).")
else:
    selected_images = all_image_files
    print(f"Using all {len(selected_images)} available images for testing.")


print(f"\nTesting against all labels: {text_labels}")
print(f"Primary Positive Target: '{POSITIVE_TARGET_PROMPT}' (Threshold: {POSITIVE_THRESHOLD})")
print(f"Exclusion Prompts: {EXCLUSION_PROMPTS} (Threshold: {EXCLUSION_THRESHOLD})")


# Preprocess text labels
# text_inputs = clip.tokenize(text_labels).to(device) # Old way for OpenAI clip
text_inputs = tokenizer(text_labels).to(device) # New way for open_clip

# Get indices for quick lookup
positive_target_idx = text_labels.index(POSITIVE_TARGET_PROMPT)
exclusion_indices = [text_labels.index(p) for p in EXCLUSION_PROMPTS if p in text_labels]


# Process images and filter
results_summary = {}
filtered_images_final = []

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for image_path in selected_images:
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probabilities = similarity[0].cpu().numpy()

            current_image_results = {label: float(prob) for label, prob in zip(text_labels, probabilities)}
            results_summary[os.path.basename(image_path)] = current_image_results

            print(f"\n--- Results for {os.path.basename(image_path)} ---")
            for i, label in enumerate(text_labels):
                print(f"{label}: {probabilities[i]*100:.2f}%")

            top_prob_index = probabilities.argmax()
            print(f"==> Overall Best match: {text_labels[top_prob_index]} ({probabilities[top_prob_index]*100:.2f}%)")

            # --- Multi-stage filtering logic ---
            is_positive_target = probabilities[positive_target_idx] >= POSITIVE_THRESHOLD
            is_excluded = False
            excluded_by_prompt = None

            for ex_idx in exclusion_indices:
                if probabilities[ex_idx] >= EXCLUSION_THRESHOLD:
                    is_excluded = True
                    excluded_by_prompt = text_labels[ex_idx]
                    break

            print(f"    Prob for '{POSITIVE_TARGET_PROMPT}': {probabilities[positive_target_idx]*100:.2f}% (Threshold: {POSITIVE_THRESHOLD*100:.0f}%) -> {'MET' if is_positive_target else 'NOT MET'}")
            if is_excluded:
                 print(f"    Prob for EXCLUSION '{excluded_by_prompt}': {probabilities[text_labels.index(excluded_by_prompt)]*100:.2f}% (Threshold: {EXCLUSION_THRESHOLD*100:.0f}%) -> EXCLUDED")
            else:
                 print(f"    No exclusion criteria met (Threshold: {EXCLUSION_THRESHOLD*100:.0f}%)")


            if is_positive_target and not is_excluded:
                filtered_images_final.append(image_path)
                print("    ==> Final filtering passed (Image meets positive target and no exclusions)")
            else:
                reason = "positive target not met" if not is_positive_target else f"excluded by '{excluded_by_prompt}'"
                print(f"    ==> Final filtering failed (Reason: {reason})")


        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

# Output filtered results
print("\n--- Filtering Complete ---")
if filtered_images_final:
    print(f"\nImages meeting ALL criteria (Positive Target AND NOT Excluded):")
    for img_path in filtered_images_final:
        print(os.path.basename(img_path))
else:
    print(f"No images met all filtering criteria with the current thresholds.")
    print("Consider adjusting POSITIVE_THRESHOLD, EXCLUSION_THRESHOLD, or reviewing prompts.")

print(f"\nTotal selected images processed: {len(selected_images)}")
print(f"Total images matching all criteria: {len(filtered_images_final)}")

# Save the full results summary
os.makedirs(Results_dir, exist_ok=True) # Ensure Results_dir exists
summary_file_path = os.path.join(Results_dir, f"classification_summary_{Name}.json")
with open(summary_file_path, "w") as f:
   json.dump(results_summary, f, indent=4)
print(f"\nFull classification results summary saved to {summary_file_path}")

# Copy filtered images
if CLIP_Filter and filtered_images_final:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    copied_count = 0
    for img_path in filtered_images_final:
        try:
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, os.path.basename(img_path)))
            copied_count +=1
        except Exception as e:
            print(f"Error copying {img_path} to {OUTPUT_DIR}: {e}")
    print(f"\n{copied_count} Filtered images copied to '{OUTPUT_DIR}/'")
elif CLIP_Filter and not filtered_images_final:
    print("\nNo images to copy as filter criteria were not met.")