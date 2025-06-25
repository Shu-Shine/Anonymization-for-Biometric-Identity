import cv2
import numpy as np
import os
import time

def watershed(INPUT_FOLDER, OUTPUT_FOLDER, BINARY_THRESHOLD_VALUE = 150, SURE_FG_THRESHOLD_RATIO = 0.6):
    # --- Configuration ---
    # INPUT_FOLDER = '/home/woody/iwso/iwso183h/image_anonymization/processed_watershed5'
    # OUTPUT_FOLDER = 'processed_watershed6'

    # Lower value for dimmer whites, higher for brighter whites.
    # BINARY_THRESHOLD_VALUE = 150

    # Lower this if the object is thin; raise it for large, solid objects.
    # SURE_FG_THRESHOLD_RATIO = 0.5

    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: '{OUTPUT_FOLDER}'")

    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Input directory not found at '{INPUT_FOLDER}'")
        exit()


    try:
        image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)]
        print(f"Found {len(image_files)} images to process in '{INPUT_FOLDER}'.")
    except FileNotFoundError:
        exit()

    if not image_files:
        print("No images found in the input folder. Exiting.")
        exit()


    for filename in image_files:
        try:
            image_path = os.path.join(INPUT_FOLDER, filename)

            base_name, ext = os.path.splitext(filename)
            # output_overlay_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_overlay{ext}")
            output_overlay_path = os.path.join(OUTPUT_FOLDER, filename)
            # output_mask_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_mask{ext}")
            print(f"\n--- Processing: {filename} ---")

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image '{filename}'. Skipping.")
                continue

            # Pre-processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, BINARY_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

            # Noise Removal (Morphological Operations)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Finding Sure Background and Foreground
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

            # Check if any foreground was found at all
            if dist_transform.max() == 0:
                print("Warning: No foreground object found after thresholding. Skipping watershed.")
                # Save the original image
                cv2.imwrite(output_overlay_path, image)
                # cv2.imwrite(output_mask_path, np.zeros_like(gray))
                continue

            ret, sure_fg = cv2.threshold(dist_transform, SURE_FG_THRESHOLD_RATIO * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

            unknown = cv2.subtract(sure_bg, sure_fg)

            # Create Markers for Watershed
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            cv2.watershed(image, markers)

            # The detected regions will have labels > 1 (background is 1, boundary is -1)
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[markers > 1] = 255  # Make detected object white

            # Create the hard overlay image
            overlay_image = image.copy()
            overlay_color = [0, 0, 0]  # [255, 0, 0] Blue
            overlay_image[mask == 255] = overlay_color

            cv2.imwrite(output_overlay_path, overlay_image)
            print(f"Successfully saved overlay to '{output_overlay_path}'")

        except Exception as e:
            print(f"!!!!!! An unexpected error occurred while processing {filename}: {e} !!!!!!")
            continue

    print("\n--- Watershed Processing Finished ---")

if __name__ == "__main__":

    INPUT_FOLDER = '/home/woody/iwso/iwso183h/output/CLIP_filter2/filtered_images'
    OUTPUT_FOLDER = 'processed_watershed'

    watershed(INPUT_FOLDER, OUTPUT_FOLDER, SURE_FG_THRESHOLD_RATIO = 0.6)