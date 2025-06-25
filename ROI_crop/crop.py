# tensorflow
import os
import cv2
import numpy as np
import pylab as plt
import sys

# Add 'Deepskin-main' directory to sys.path
# Ensure this path is correct relative to where you run the script.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Deepskin-main'))
try:
    from deepskin import wound_segmentation
except ImportError:
    print("Error: Could not import 'wound_segmentation'.")
    print("Please ensure the 'Deepskin-main' directory is correctly located and in the Python path.")
    sys.exit(1)


def crop_and_save_individual_wounds(img_path, base_output_path, margin=50, min_area=100):
    """
    Detects all wounds in an image, and saves each individual wound as a separate cropped image.

    Args:
        img_path (str): Path to the input image.
        base_output_path (str): The base path for saving files (e.g., 'output/image1').
                               The function will append '_wound_N.png' to this.
        margin (int): The pixel margin to add around each wound's bounding box.
        min_area (int): The minimum pixel area for a detected component to be considered a wound.
    """
    print(f"\nProcessing image: {img_path}")

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Error: Could not load image at {img_path}")
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    try:
        # Get the segmentation mask from the deep learning model
        segmentation = wound_segmentation(img=rgb)
        wound_mask, _, _ = cv2.split(segmentation)
    except Exception as e:
        print(f"Error during wound segmentation: {e}")
        return

    # --- NEW LOGIC: Use Connected Components to find each individual wound ---
    # This function finds all separate white "blobs" in the mask
    # It returns the number of labels, a matrix with each blob labeled, and stats for each blob.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wound_mask, connectivity=8)

    # We start from label 1, because label 0 is the background.
    wound_count = 0
    if num_labels > 1:
        print(f"Found {num_labels - 1} potential wound component(s).")
        for i in range(1, num_labels):
            # Get the statistics for the current component
            area = stats[i, cv2.CC_STAT_AREA]

            # --- Filter out small, noisy components ---
            if area < min_area:
                print(f"  - Skipping component {i} (area: {area}px) - smaller than min_area ({min_area}px).")
                continue

            wound_count += 1
            print(f"  - Processing wound #{wound_count} (area: {area}px).")

            # Get the bounding box for this specific component
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Calculate the crop coordinates with the margin
            crop_x1 = max(0, x - margin)
            crop_y1 = max(0, y - margin)
            crop_x2 = min(w, x + width + margin)
            crop_y2 = min(h, y + height + margin)

            # Crop the original image using these coordinates
            cropped_rgb = rgb[crop_y1:crop_y2, crop_x1:crop_x2]

            # Generate a unique output filename for this specific wound
            output_filename = f"{base_output_path}_wound_{wound_count}.png"

            # Save the cropped image
            plt.figure(figsize=(7, 7))
            plt.imshow(cropped_rgb)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_filename)
            plt.close()
            print(f"    Saved cropped wound to {output_filename}")

    if wound_count == 0:
        print("Warning: No wounds detected (or all were smaller than min_area).")


def batch_process_wounds(input_dir, output_dir, margin=50, min_area=100):
    """
    Processes a whole directory of images to find and save individual wounds.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No images found in '{input_dir}'. Please check the path.")
        return

    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        name, _ = os.path.splitext(filename)

        # Create a base name for the output files for this image
        base_output_path = os.path.join(output_dir, name)

        # Call the core function to process one image
        crop_and_save_individual_wounds(
            img_path,
            base_output_path=base_output_path,
            margin=margin,
            min_area=min_area
        )


if __name__ == "__main__":
    # --- Configuration ---
    input_folder = "/home/woody/iwso/iwso183h/image_anonymization/output/CLIP_filter3/filtered_images"
    output_folder = "../output/Crop/individual_wounds"

    # The pixel margin to add around each detected wound
    CROP_MARGIN = 50

    # The minimum size in pixels for a detection to be saved.
    # IMPORTANT: Adjust this to avoid saving tiny noise specks.
    MINIMUM_WOUND_AREA = 200

    batch_process_wounds(
        input_dir=input_folder,
        output_dir=output_folder,
        margin=CROP_MARGIN,
        min_area=MINIMUM_WOUND_AREA
    )