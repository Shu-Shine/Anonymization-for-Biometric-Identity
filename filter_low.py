import torch
import clip
from PIL import Image
import os
import json
import shutil
import random
from pathlib import Path
import datetime


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
IMAGE_DIR = "/home/woody/iwso/iwso183h/data/filtered_images_6"
# IMAGE_DIR = "test_noise"
# IMAGE_DIR = "/home/woody/iwso/iwso183h/data/ChroWound_process/filtered_images_5/rejected_images"
CLIP_Filter = True  # Master switch to enable/disable saving any images

Results_dir = increment_path("output", name="CLIP_filter")
# Results_dir = increment_path("data/ChroWound_process", name="filtered_images_5")
# Results_dir.mkdir(parents=True, exist_ok=True)

# --- Resolution Control ---
SAVE_LOW_RESOLUTION = True  # todo: True: save smaller images, False: save originals.
LOW_RES_SIZE = (256, 256)  # for low-resolution images.
# --- END NEW ---

Name = "filtered_images"
Rejected_Name = "rejected_images"
OUTPUT_DIR = Results_dir / Name
REJECTED_OUTPUT_DIR = Results_dir / Rejected_Name

SUBSET_SIZE = 50  # Set to None or a larger number to use all images

# --- CLIP Model Configuration ---
CLIP_MODEL_NAME = 'ViT-B/16'

# --- Prompt Configuration ---
POSITIVE_TARGET_PROMPT = "a photo of a skin wound"
POSITIVE_THRESHOLD = 0.01

EXCLUSION_PROMPTS_AND_THRESHOLDS = [
    # {"prompt": "a photo of a bandaged body", "threshold": 0.50},
    {"prompt": "a photo of blue lamp examination", "threshold": 0.50},
    # {"prompt": "a photo of a skin grafting site wound", "threshold": 0.50},  #0.45
    # {"prompt": "a photo of a skin grafting", "threshold": 0.78},  #0.45
    {"prompt": "a photo of wound treatment", "threshold": 0.31},  # ori 0.31, no higher than 0.35
    {"prompt": "a photo of post treatment wound", "threshold": 0.80},  # 0.83 ori 0.73
    # {"prompt": "a photo of surgery", "threshold": 0.21},
    {"prompt": "a photo of body without wound", "threshold": 0.30},
    {"prompt": "a photo of wound dressing", "threshold": 0.41},  #0.43
    # {"prompt": "a photo of application of ointment to the wound", "threshold": 0.25},
    # {"prompt": "a photo of wound with ruler", "threshold": 0.99},
]

# --- End of Configuration ---

# Setup derived configurations
all_exclusion_prompt_strings = [item["prompt"] for item in EXCLUSION_PROMPTS_AND_THRESHOLDS]
text_labels = [POSITIVE_TARGET_PROMPT] + all_exclusion_prompt_strings + [
        # Add other relevant general categories if needed for broader classification output
        # "a photo of total blue Wood's lamp examination",
        # "a photo of Skin slice specimens",
        # "a photo of bandaged body part",
        "a photo of wound with ruler"
    ]
if len(text_labels) != len(set(text_labels)):
    print("Warning: Duplicate prompts found. De-duplicating...")
    text_labels = sorted(list(set(text_labels)))

# Load the CLIP model and preprocessor
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Attempting to load OpenAI CLIP model: {CLIP_MODEL_NAME} on {device}")
try:
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    model.eval()
    print(f"Model '{CLIP_MODEL_NAME}' loaded successfully on {device}")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit()

# Prepare image files
all_image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
if not all_image_files:
    print(f"Error: No images found in '{IMAGE_DIR}'.")
    exit()
print(f"Found {len(all_image_files)} images in '{IMAGE_DIR}'.")

# Select a subset of images
if SUBSET_SIZE is not None and len(all_image_files) > SUBSET_SIZE:
    selected_images = random.sample(all_image_files, SUBSET_SIZE)
else:
    selected_images = all_image_files
print(f"Processing {len(selected_images)} images.")

# --- Create output directories before the loop ---
if CLIP_Filter:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REJECTED_OUTPUT_DIR, exist_ok=True)
    print(f"Saving accepted images to: {OUTPUT_DIR}")
    print(f"Saving rejected images to: {REJECTED_OUTPUT_DIR}")
    if SAVE_LOW_RESOLUTION:
        print(f"Saving mode: Low Resolution (max size: {LOW_RES_SIZE})")
    else:
        print("Saving mode: Original Resolution")

# Preprocess text labels
text_inputs = clip.tokenize(text_labels).to(device)
positive_target_idx = text_labels.index(POSITIVE_TARGET_PROMPT)

# Process images and filter
results_summary = {}
accepted_count = 0
rejected_count = 0

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for image_path in selected_images:
        image_filename = os.path.basename(image_path)
        try:
            # Note: We keep the 'image' object in memory for potential resizing
            image = Image.open(image_path).convert("RGB")

            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probabilities = similarity[0].cpu().numpy()

            current_image_results = {label: float(prob) for label, prob in zip(text_labels, probabilities)}
            results_summary[image_filename] = current_image_results

            print(f"\n--- Results for {image_filename} ---")
            top_prob_index = probabilities.argmax()
            print(f"==> Overall Best match: {text_labels[top_prob_index]} ({probabilities[top_prob_index] * 100:.2f}%)")

            # --- Multi-stage filtering logic ---
            is_positive_target = probabilities[positive_target_idx] >= POSITIVE_THRESHOLD

            is_excluded = False
            excluded_by_prompt = None
            for ex_definition in EXCLUSION_PROMPTS_AND_THRESHOLDS:
                prompt_str = ex_definition["prompt"]
                threshold_val = ex_definition["threshold"]
                prompt_idx = text_labels.index(prompt_str)
                if probabilities[prompt_idx] >= threshold_val:
                    is_excluded = True
                    excluded_by_prompt = prompt_str
                    break

            # --- Decision and Immediate Saving ---
            destination_folder = None
            if is_positive_target and not is_excluded:
                print("    ==> Final filtering passed (Image ACCEPTED)")
                destination_folder = OUTPUT_DIR
                accepted_count += 1
            else:
                reason = "positive target not met" if not is_positive_target else f"excluded by '{excluded_by_prompt}'"
                print(f"    ==> Final filtering failed (Image REJECTED - Reason: {reason})")
                destination_folder = REJECTED_OUTPUT_DIR
                rejected_count += 1

            # --- Save the image to the determined folder ---
            if CLIP_Filter and destination_folder:
                dest_path = os.path.join(destination_folder, image_filename)
                try:
                    if SAVE_LOW_RESOLUTION:
                        # Create a resized version and save it
                        # .thumbnail() preserves aspect ratio, which is better than .resize()
                        low_res_image = image.copy()
                        low_res_image.thumbnail(LOW_RES_SIZE, Image.Resampling.LANCZOS)
                        low_res_image.save(dest_path, "JPEG", quality=85)  # Save as JPEG with good quality
                    else:
                        # Copy the original, full-resolution file
                        shutil.copy(image_path, dest_path)
                except Exception as e:
                    print(f"    [ERROR] Could not save image {image_filename} to {destination_folder}: {e}")

        except Exception as e:
            print(f"    [ERROR] Critical error processing {image_filename}: {e}")

# --- Final Summary ---
print("\n--- Filtering Complete ---")
print(f"\nTotal selected images processed: {len(selected_images)}")
print(f"Total images accepted: {accepted_count}")
print(f"Total images rejected: {rejected_count}")

# Save the full classification results summary
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
summary_filename = f"classification_summary_{timestamp}.json"
summary_file_path = os.path.join(Results_dir, summary_filename)
with open(summary_file_path, "w") as f:
    json.dump(results_summary, f, indent=4)
print(f"\nFull classification results summary saved to {summary_file_path}")

if CLIP_Filter:
    print(f"\n{accepted_count} Accepted images saved in '{OUTPUT_DIR}/'")
    print(f"{rejected_count} Rejected images saved in '{REJECTED_OUTPUT_DIR}/'")