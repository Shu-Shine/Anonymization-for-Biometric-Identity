import sys
from pathlib import Path
import os

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[0]
YOLOV5_DIR = PROJECT_ROOT / 'yolov5-master'
DSFD_DIR = PROJECT_ROOT / 'DSFD-Pytorch-Inference-master'

# Add the yolov5-master directory to the Python path
if str(YOLOV5_DIR) not in sys.path:
    sys.path.append(str(YOLOV5_DIR))

# --- Import the tattoo detection function ---
try:
    from tattoo_detector import detect_tattoos_in_images
    print("Successfully imported detect_tattoos_in_images.")
except ImportError as e:
    print(f"ERROR: Could not import from 'yolov5-master/tattoo_detector.py'.")
    print(f"Ensure the file exists at {YOLOV5_DIR / 'tattoo_detector.py'} "
          f"and the yolov5 environment is active.")
    print(f"Import Error: {e}")
    sys.exit(1)

# Add the DSFD directory to the Python path
if str(DSFD_DIR) not in sys.path:
   sys.path.insert(0, str(DSFD_DIR))

# --- Import the face detection function ---
try:
    from face_detector import detect_faces_in_images
except ImportError:
    print("Error: Could not import detect_faces_in_images.")
    print("Ensure face_processor.py is in your Python path or current directory.")
    sys.exit(1)


# --- Main Application Logic ---
if __name__ == "__main__":
    print("Starting main application script.")

    # Define the input image folder
    # input_image_folder = PROJECT_ROOT / 'data' / 'BIVTatt-Dataset-master' / 'images'  # e.g., ./data/input_images
    input_image_folder = PROJECT_ROOT / 'images'

    if not input_image_folder.exists():
        print(f"Input folder does not exist: {input_image_folder}")

    # --- Tattoo Detection ---
    print(f"\n--- Running tattoo detection on folder: {input_image_folder} ---")
    output_location_tattoo = PROJECT_ROOT / 'output' / 'tattoo_detection_results'
    try:
        results_dir_1 = detect_tattoos_in_images(
            source=str(input_image_folder),  # Pass path as string
            # weights=str(YOLOV5_DIR / 'your_custom_weights.pt'), # Optional: Override weights
            project=str(output_location_tattoo),  # 'output/detection_results'
            name='run_folder',   # Name of the subfolder for each run
            conf_thres=0.4,
            create_dataset_with_tattoo=True,    # Create dataset with tattoo
            create_dataset_without_tattoo=False,
            nosave=True  # Not save annotated images directly to output folder
        )
        print(f"Tattoo detection finished. Results saved to: {results_dir_1}")

    except FileNotFoundError as e:
         print(f"ERROR during tattoo detection: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during tattoo detection: {e}")


    # --- Face Detection ---
    print(f"\n--- Running face detection on folder: {input_image_folder} ---")
    output_location_face = PROJECT_ROOT / 'output' / 'face_detection_results'
    try:
        results_directory = detect_faces_in_images(
            image_source=input_image_folder,
            results_dir=output_location_face,
            name='run_folder',   # Name of the subfolder for each run
            create_dataset_with_face=True,   # Create dataset with face
            create_dataset_without_face=False,
            draw_boxes_on_saved=True  # Draw boxes, not just copy
        )
        print(f"Face detection complete. Results are in: {results_directory}")

    except FileNotFoundError as e:
        print(f"Error: Input path not found - {e}")
    except Exception as e:
        print(f"An error occurred during face detection: {e}")

    print("\nMain application script finished.")