import easyocr
import cv2
import numpy as np
import os
import time


def easyocr_processor(INPUT_FOLDER, OUTPUT_FOLDER, MAX_DIM = 1000):
    # --- Configuration ---
    # INPUT_FOLDER = '/home/woody/iwso/iwso183h/image_anonymization/output/CLIP_filter/filtered_images'
    # OUTPUT_FOLDER = 'processed_images2'

    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # Resizing setting (for large images)
    # MAX_DIM = 1000  # Max height or width for resizing


    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: '{OUTPUT_FOLDER}'")

    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Input directory not found at '{INPUT_FOLDER}'")
        exit()


    print("Loading EasyOCR model...")
    start_time = time.time()
    try:
        # gpu=False
        reader = easyocr.Reader(['en'], gpu=False)
        print(f"EasyOCR model loaded successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"FATAL: Error loading EasyOCR model: {e}")
        exit()


    try:
        image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(IMAGE_EXTENSIONS)]
        print(f"Found {len(image_files)} images to process in '{INPUT_FOLDER}'.")
    except FileNotFoundError:
        exit()

    if not image_files:
        print("No images found in the input folder. Exiting.")
        exit()

    # Process each image file
    for filename in image_files:
        try:
            image_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            print(f"\n--- Processing: {filename} ---")

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image '{filename}'. Skipping.")
                continue

            original_shape = image.shape

            # --- Resize if necessary ---
            height, width, _ = image.shape
            if height > MAX_DIM or width > MAX_DIM:
                if height > width:
                    new_height = MAX_DIM
                    aspect_ratio = width / height
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = MAX_DIM
                    aspect_ratio = height / width
                    new_height = int(new_width * aspect_ratio)

                print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
                image_for_processing = cv2.resize(image, (new_width, new_height))
            else:
                image_for_processing = image.copy()

            # --- Detect Text ---
            print("Detecting text...")
            proc_start_time = time.time()
            results = reader.readtext(image_for_processing)
            print(f"Detection finished in {time.time() - proc_start_time:.2f} seconds. Found {len(results)} text boxes.")

            # --- Draw Hard Overlay ---
            overlay_image = image_for_processing.copy()
            overlay_color = (255, 255, 255)  # white

            if not results:
                print("No text detected in this image.")
            else:
                for (bbox, text, prob) in results:
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.fillPoly(overlay_image, [pts], overlay_color)
                    # print(f"  - Detected: '{text}' (Confidence: {prob:.2f})")

            cv2.imwrite(output_path, overlay_image)
            print(f"Successfully saved result to '{output_path}'")

        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e} ")
            continue

    print("\n--- Easyocr Processing Script Finished ---")