import torch
import clip
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
# Set the input image directory
IMAGE_DIR = "../images"  # Directory containing images to filter
TARGET_LABEL = "a photo of wound, without exposed genitalia, anus, perineal region, Buttocks or Female breasts, without face, without tattoo or any recognizable features or private parts of the human body"

# Set to True to enable saving filtered images
CLIP_Filter = False
Results_dir = "../output/CLIP_filter"
Name = "filtered_images"
OUTPUT_DIR = increment_path(Results_dir, name=Name)

# Set the number of images to randomly select for testing
SUBSET_SIZE = 300

# Set the threshold for classification
PROBABILITY_THRESHOLD = 0.5
MODEL_NAME = "ViT-B/32"  # other options: "ViT-B/16", "ViT-L/14"
# --- End of Configuration ---

# Load the CLIP model and preprocessor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(MODEL_NAME, device=device)
print(f"Model '{MODEL_NAME}' loaded on {device}")

# Prepare image files
image_files = []
if os.path.exists(IMAGE_DIR):
    for f_name in os.listdir(IMAGE_DIR):
        if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_files.append(os.path.join(IMAGE_DIR, f_name))
else:
    print(f"Error: Image directory '{IMAGE_DIR}' not found.")
    exit()

if not image_files:
    print(f"No images found in '{IMAGE_DIR}'.")
    exit()
print(f"Found {len(image_files)} images to test in '{IMAGE_DIR}'.")

# Select a subset of images for testing
if len(image_files) < SUBSET_SIZE:
    print(f"Warning: Only {len(image_files)} images found, using all available images for testing.")
    selected_images = image_files  # Use all images
else:
    selected_images = random.sample(image_files, SUBSET_SIZE)
print(f"\n Selected {len(selected_images)} images for testing.")

# Define text labels
text_labels = [
    "a photo of wound, without exposed genitalia, anus, perineal region, Buttocks or Female breasts, without face, without tattoo or any recognizable features or private parts of the human body",
    "a photo of exposed genitalia, anus, perineal region, Buttocks, Female breasts or other Sensitive Private Body Parts",
    "a photo of a face or other Biometric Identity",
    "a photo of Tattoo, scars, birthmarks, moles, or other skin markings",
    "a photo of total blue Wood's lamp examination",
    "a photo of ID tag",
    "a photo of Skin slice specimens",
    "a photo of bandaged body",
    "a photo of skin without wound",
    # cases for wound on private parts
    "a photo of skin wound on exposed genitalia, anus, perineal region, Buttocks or Female breasts area",
    # "a photo of wound, with exposed genitalia, anus, perineal region, Buttocks Female breasts, or face, or tattoo or any recognizable features or private parts of the human body",

]

print(f"\nTesting against labels: {text_labels}")


# Check if target label is in the list and get its index
try:
    target_label_index = text_labels.index(TARGET_LABEL)
except ValueError:
    print(f"\nError: Target label '{TARGET_LABEL}' not found in the predefined text_labels list.")
    exit()

# Preprocess text labels
text_inputs = clip.tokenize(text_labels).to(device)

# Process images and filter
results_summary = {} # To store probabilities for all images for potential review
filtered_lesion_images = []

with torch.no_grad(): # Disable gradient calculations
    # Encode text features once
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True) # Normalize features

    for image_path in selected_images:
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Encode image features
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize features

            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probabilities = similarity[0].cpu().numpy() # Get the probabilities

            # Store all probabilities for this image
            current_image_results = {label: float(prob) for label, prob in zip(text_labels, probabilities)}
            results_summary[os.path.basename(image_path)] = current_image_results

            print(f"\n--- Results for {os.path.basename(image_path)} ---")
            for i, label in enumerate(text_labels):
                print(f"{label}: {probabilities[i]*100:.2f}%")

            # Get the overall top prediction for informational purposes
            top_prob_index = probabilities.argmax()
            print(f"==> Overall Best match: {text_labels[top_prob_index]} ({probabilities[top_prob_index]*100:.2f}%)")

            # Check against the target label and threshold
            probability_for_target_label = probabilities[target_label_index]
            print(f"==> Probability for '{TARGET_LABEL}': {probability_for_target_label*100:.2f}%")

            if probability_for_target_label >= PROBABILITY_THRESHOLD:
                filtered_lesion_images.append(image_path)
                print(f"    -> CLASSIFIED as '{TARGET_LABEL}' (threshold {PROBABILITY_THRESHOLD*100:.0f}%)")
            else:
                print(f"    -> NOT CLASSIFIED as '{TARGET_LABEL}' (threshold {PROBABILITY_THRESHOLD*100:.0f}%)")


        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

# Output filtered results
print("\n--- Filtering Complete ---")
if filtered_lesion_images:
    print(f"\nImages classified as '{TARGET_LABEL}' (with probability >= {PROBABILITY_THRESHOLD*100:.0f}%):")
    for img_path in filtered_lesion_images:
        print(os.path.basename(img_path))
else:
    print(f"No images were classified as '{TARGET_LABEL}' with the current threshold of {PROBABILITY_THRESHOLD*100:.0f}%.")
    print("Consider adjusting the PROBABILITY_THRESHOLD or reviewing your text labels and images.")

# print(f"\nTotal images processed: {len(image_files)}")
print(f"\nTotal selected images processed: {len(selected_images)}")
print(f"Total images matching target_label: {len(filtered_lesion_images)}")

# Save the full results summary to a file for detailed review
with open("full_clip_classification_results.json", "w") as f:
   json.dump(results_summary, f, indent=4)
print("\nFull classification results summary saved to full_clip_classification_results.json")

# Copy filtered images to a new directory
if CLIP_Filter:
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the folder if it doesn't exist

    for img_path in filtered_lesion_images:
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, os.path.basename(img_path)))

    print(f"\nFiltered images copied to '{OUTPUT_DIR}/'")