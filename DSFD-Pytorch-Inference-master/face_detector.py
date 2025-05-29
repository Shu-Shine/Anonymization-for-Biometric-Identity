import glob
import os
import cv2
import time
import face_detection
import csv
import shutil # For copying files
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# helper function to increment path
def increment_path(base_dir, name='exp'):
    base = Path(base_dir)
    path = base / name

    if not path.exists():
        return path.resolve()
    else:
        # 如果 name 已存在，从 name2 开始找
        n = 2
        while (base / f"{name}{n}").exists():
            n += 1
        return (base / f"{name}{n}").resolve()


def detect_faces_in_images(
    image_source: str,                     # Directory or single image file path
    results_dir: str,                      # Directory to save all results
    name: str = "exp",                     # Name of the results directory
    create_dataset_with_face: bool = True, # Save images where faces are detected
    create_dataset_without_face: bool = False,# Save images where no faces are detected
    draw_boxes_on_saved: bool = True,      # Draw detection boxes on the saved images
    detector_model: str = "DSFDDetector",  # Model name for face_detection library
    max_resolution: int = 1080             # Max resolution for the detector
) -> Path:
    """
    Detects faces in images from a source directory or file, saves results,
    and optionally creates datasets of images with/without faces.

    Args:
        image_source (str): Path to the input directory containing images or a single image file.
        results_dir (str): Path to the directory where results (CSVs, output images) will be saved.
        create_dataset_with_face (bool): If True, saves images where faces were detected
                                         to a 'faces' subdirectory.
        create_dataset_without_face (bool): If True, saves images where no faces were detected
                                            to a 'no_faces' subdirectory.
        draw_boxes_on_saved (bool): If True, draws bounding boxes on the images saved
                                    in the 'faces' subdirectory. If False, original images are copied.
        detector_model (str): The face detector model to use (e.g., "DSFDDetector").
        max_resolution (int): Maximum resolution constraint for the detector.

    Returns:
        pathlib.Path: The absolute path to the results directory.

    Raises:
        FileNotFoundError: If the image_source path does not exist.
        ImportError: If the 'face_detection' library is not found.
        Exception: Catches potential errors during detection or file operations.
    """

    # --- Path Setup ---
    source_path = Path(image_source).resolve()
    # results_path = Path(results_dir).resolve()
    results_path = increment_path(results_dir, name=name) # Create a unique results directory

    if not source_path.exists():
        raise FileNotFoundError(f"Image source not found: {source_path}")

    # Define specific output directories
    faces_output_dir = results_path / "faces"
    no_faces_output_dir = results_path / "no_faces"

    # Create output directories
    results_path.mkdir(parents=True, exist_ok=True)
    if create_dataset_with_face:
        faces_output_dir.mkdir(exist_ok=True)
    if create_dataset_without_face:
        no_faces_output_dir.mkdir(exist_ok=True)

    # --- Find Images ---
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.jfif", "*.bmp", "*.webp"]
    impaths = []
    if source_path.is_file():
        if source_path.suffix.lower().lstrip('.') in [ext.lstrip("*.") for ext in image_extensions]:
             impaths = [source_path]
        else:
             logging.warning(f"Source file is not a supported image type: {source_path}")
    elif source_path.is_dir():
        for ext in image_extensions:
            impaths.extend(source_path.glob(ext))
    else:
         raise ValueError(f"Invalid image source path: {source_path}")

    if not impaths:
        logging.warning(f"No image files found in {source_path}")
        return results_path # Return early if no images

    logging.info(f"Found {len(impaths)} image(s) to process.")

    # --- Initialize Detector ---
    try:
        detector = face_detection.build_detector(
            detector_model,
            max_resolution=max_resolution
        )
        logging.info(f"Initialized detector: {detector_model}")
    except Exception as e:
        logging.error(f"Failed to build face detector '{detector_model}': {e}")
        raise RuntimeError(f"Failed to build face detector: {e}") from e


    # --- CSV Setup ---
    main_csv_path = results_path / "face_detection_results.csv"
    faces_csv_path = results_path / "images_with_faces.csv"
    no_faces_csv_path = results_path / "images_without_faces.csv"

    # Using 'with open' ensures files are closed automatically
    try:
        with open(main_csv_path, 'w', newline='', encoding='utf-8') as main_csv_file, \
             open(faces_csv_path, 'w', newline='', encoding='utf-8') as faces_csv_file, \
             open(no_faces_csv_path, 'w', newline='', encoding='utf-8') as no_faces_csv_file:

            main_writer = csv.writer(main_csv_file)
            faces_writer = csv.writer(faces_csv_file)
            no_faces_writer = csv.writer(no_faces_csv_file)

            # Write headers
            main_writer.writerow(["filename", "prediction (face detected)"])
            faces_writer.writerow(["filename"])
            no_faces_writer.writerow(["filename"])

            # --- Process Images ---
            results_summary = []
            imgs_with_faces_list = []
            imgs_without_faces_list = []
            start_time = time.time()

            for i, impath in enumerate(impaths, 1):
                filename = impath.name
                logging.info(f"Processing [{i}/{len(impaths)}]: {filename}")
                im = cv2.imread(str(impath)) # Read image

                if im is None:
                    logging.error(f"Could not read image {impath}")
                    results_summary.append([filename, "Error: Could not read"])
                    continue # Skip to next image

                faces_found = False
                detection_error = None
                dets = []

                try:
                    # Detect faces (detector expects RGB, OpenCV loads BGR)
                    dets = detector.detect(im[:, :, ::-1])[:, :4] # Get bounding boxes
                    faces_found = len(dets) > 0
                except Exception as e:
                    logging.error(f"Error during face detection on {filename}: {e}")
                    detection_error = f"Error: {e}"

                # Record main result
                if detection_error:
                    results_summary.append([filename, detection_error])
                else:
                    results_summary.append([filename, str(faces_found)]) # Store as string "True" or "False"

                # Handle image saving/copying and specific CSV lists
                if faces_found:
                    imgs_with_faces_list.append([filename])
                    if create_dataset_with_face:
                        dest_path = faces_output_dir / filename
                        if draw_boxes_on_saved:
                            # Draw boxes on a copy
                            im_copy = im.copy()
                            for bbox in dets:
                                x0, y0, x1, y1 = [int(_) for _ in bbox]
                                cv2.rectangle(im_copy, (x0, y0), (x1, y1), (0, 0, 255), 2)
                            cv2.imwrite(str(dest_path), im_copy)
                        else:
                            # Copy original file
                            try:
                                shutil.copy2(impath, dest_path) # copy2 preserves metadata
                            except Exception as copy_err:
                                logging.warning(f"Could not copy {filename} to {dest_path}: {copy_err}")
                else:
                    # Only add to this list if no error occurred during detection
                    if not detection_error:
                        imgs_without_faces_list.append([filename])

                    if create_dataset_without_face:
                         dest_path = no_faces_output_dir / filename
                         # Always copy original if no faces were found
                         try:
                             shutil.copy2(impath, dest_path)
                         except Exception as copy_err:
                              logging.warning(f"Could not copy {filename} to {dest_path}: {copy_err}")


            # --- Write accumulated CSV data after loop ---
            main_writer.writerows(results_summary)
            faces_writer.writerows(imgs_with_faces_list)
            no_faces_writer.writerows(imgs_without_faces_list)

            end_time = time.time()
            total_processing_time = end_time - start_time
            logging.info(f"Total processing time: {total_processing_time:.2f} seconds for {len(impaths)} images.")

            # --- Print Summary ---
            total_images = len(impaths)
            images_processed_ok = len([row for row in results_summary if not row[1].startswith("Error:")])
            images_with_faces_count = len(imgs_with_faces_list)

            logging.info(f"Successfully processed {images_processed_ok}/{total_images} images.")
            if images_processed_ok > 0:
                 percentage_with_faces = (images_with_faces_count / images_processed_ok) * 100
                 logging.info(f"Percentage of processed images containing faces: {percentage_with_faces:.2f}% ({images_with_faces_count}/{images_processed_ok})")
            else:
                 logging.info("No images were processed successfully.")


    except IOError as e:
        logging.error(f"Error opening or writing CSV file(s) in {results_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}")
        raise # Re-raise the exception

    logging.info(f"Processing complete. Results saved in: {results_path}")
    return results_path


if __name__ == "__main__":
    print("--- Running Face Detection as a direct script for testing ---")

    # Define parameters for the test run
    test_image_dir = "../data/original/D"
    test_results_dir = "face_detection_output_test"

    # Ensure the test input directory exists
    if not Path(test_image_dir).exists():
        print(f"ERROR: Test image directory not found: {test_image_dir}")
        print("Please create it or adjust the 'test_image_dir' variable.")
    else:
        try:
            output_path = detect_faces_in_images(
                image_source=test_image_dir,
                results_dir=test_results_dir,
                create_dataset_with_face=False,      # Save images with faces
                create_dataset_without_face=False,   # Save images without faces
                draw_boxes_on_saved=True            # Draw boxes on saved images
            )
            print(f"\nTest run finished. Check results in: {output_path}")

        except FileNotFoundError as e:
            print(f"ERROR during test run: {e}")
        except RuntimeError as e:
             print(f"ERROR during test run (detector related): {e}")
        except Exception as e:
            print(f"An unexpected error occurred during the test run: {e}")

    print("--- Direct script execution finished ---")