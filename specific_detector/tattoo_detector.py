import argparse
import sys
import pathlib
import os
import csv
import shutil
from pathlib import Path

# --- Path Setup ---
# This makes paths work correctly even if the calling script is outside yolov5-master
# YOLOV5_ROOT is the directory containing this script (yolov5-master)
YOLOV5_ROOT = Path(__file__).resolve().parents[0]
# Add YOLOV5_ROOT to sys.path so its internal modules can be found
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

# Workaround for non-Windows systems if needed (often Path handles this automatically)
# Consider if you really need this line.
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

# --- YOLOv5 Imports (relative to YOLOV5_ROOT) ---
try:
    import torch
    # Assuming these utils are directly under YOLOV5_ROOT or in findable subdirs
    from utils.dataloaders import IMG_FORMATS, LoadImages # Only LoadImages needed
    from utils.general import (
        LOGGER,
        Profile,
        check_file,
        check_img_size,
        # check_imshow, # Not essential for background processing
        # check_requirements, # Call this explicitly if needed before running
        colorstr,
        cv2,
        increment_path,
        non_max_suppression,
        print_args, # Optional: for debugging args passed
        scale_boxes,
    )
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device, smart_inference_mode
    from ultralytics.utils.plotting import Annotator, colors
except ImportError as e:
    print(f"ERROR: Failed to import YOLOv5 dependencies from {YOLOV5_ROOT}.")
    print(f"Ensure '{YOLOV5_ROOT}' contains the necessary YOLOv5 code and "
          f"the required environment is active.")
    print(f"Import error: {e}")
    sys.exit(1)
# --- End Imports ---


@smart_inference_mode()
def detect_tattoos_in_images(
    source: str,                        # REQUIRED: Path to image file or directory of images
    weights: str = str(YOLOV5_ROOT / 'best.pt'), # Default weights path inside yolov5-master
    project: str = 'runs/detect_tattoos', # Default output dir relative to CWD of caller
    name: str = 'exp',                     # Default experiment name
    data: str | None = None,               # Optional: Path to data.yaml (often needed by model)
                                          # Default relative to YOLOV5_ROOT if not provided explicitly
    imgsz: tuple[int, int] = (640, 640),   # Inference size (height, width)
    conf_thres: float = 0.25,              # Confidence threshold
    iou_thres: float = 0.45,               # NMS IoU threshold
    device: str = '',                      # cuda device, i.e. 0 or 0,1,2,3 or cpu
    nosave: bool = False,                  # Do not save annotated images
    exist_ok: bool = False,                # Allow overwriting existing project/name
    classes: list[int] | None = None,      # Filter by class: e.g., [0] or [0, 2]
    create_dataset_with_tattoo: bool = False, # If True, copy images containing tattoos
    create_dataset_without_tattoo: bool = False, # If True, copy images not containing tattoos
) -> Path:
    """
    Detects tattoos in images using a YOLOv5 model.

    Args:
        source (str): Path to the input image file or directory containing images.
                      Can be relative to the calling script's location (CWD).
        weights (str): Path to the YOLOv5 model weights (.pt file).
                       Defaults to 'best.pt' inside the yolov5-master folder.
        project (str): Root directory to save results. Relative to the calling
                       script's location (CWD) unless an absolute path is given.
        name (str): Subdirectory name for this specific run within 'project'.
        data (str | None): Path to the dataset.yaml file. If None, tries to use
                           a default like 'data/coco128.yaml' relative to yolov5-master,
                           but providing it explicitly is recommended if your model
                           needs it.
        imgsz (tuple[int, int]): Inference image size (height, width).
        conf_thres (float): Confidence threshold for object detection.
        iou_thres (float): IoU threshold for Non-Maximum Suppression (NMS).
        device (str): Device to run inference on ('cpu', '0', '0,1', etc.).
        nosave (bool): If True, annotated images will not be saved.
        exist_ok (bool): If True, existing `project/name` directory can be reused.
        classes (list[int] | None): Filter detections by class index(es).
        create_dataset_with_tattoo (bool): If True, creates a subdirectory in the
                                           results folder and copies images where
                                           tattoos were detected.
        create_dataset_without_tattoo (bool): If True, creates a subdirectory and
                                              copies images where no tattoos were
                                              detected.

    Returns:
        pathlib.Path: The path to the directory where results (CSVs, copied images,
                      annotated images) were saved.
    """
    # --- Argument Processing & Path Handling ---
    source = str(source) # Ensure source is string
    source_path = Path(source)
    if not source_path.exists():
         LOGGER.error(f"Source path does not exist: {source_path.resolve()}")
         raise FileNotFoundError(f"Source path does not exist: {source_path.resolve()}")

    # Resolve project path relative to the Current Working Directory (CWD) of the caller
    project_path = Path(project).resolve()

    # Resolve weights path. If relative, assume it's relative to YOLOV5_ROOT
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = YOLOV5_ROOT / weights_path
    if not weights_path.exists():
         LOGGER.error(f"Weights file not found: {weights_path}")
         raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Resolve data path if provided. If relative, assume relative to YOLOV5_ROOT
    # Handle the default case if data is None
    if data is None:
        data_path = YOLOV5_ROOT / 'data/coco128.yaml' # Default if not specified
        LOGGER.info(f"Data path not specified, using default: {data_path}")
    else:
        data_path = Path(data)
        if not data_path.is_absolute():
            data_path = YOLOV5_ROOT / data_path

    if not data_path.exists():
         LOGGER.warning(f"{colorstr('yellow', 'WARNING ⚠️')} Data file not found: {data_path}. Model loading might fail if it relies on class names from here.")
         data_path_str = None # Pass None to backend if file doesn't exist
    else:
        data_path_str = str(data_path)


    # Output directory setup
    save_dir = increment_path(project_path / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_img = not nosave

    # Dataset copy directories
    with_tattoo_dir = None
    without_tattoo_dir = None
    if create_dataset_with_tattoo:
        with_tattoo_dir = save_dir / 'tattoo'
        with_tattoo_dir.mkdir(exist_ok=True) # No parents needed, save_dir exists
        LOGGER.info(f"Will copy images with tattoos to: {with_tattoo_dir.resolve()}")
    if create_dataset_without_tattoo:
        without_tattoo_dir = save_dir / 'no_tattoo'
        without_tattoo_dir.mkdir(exist_ok=True)
        LOGGER.info(f"Will copy images without tattoos to: {without_tattoo_dir.resolve()}")

    # --- CSV Setup ---
    csv_path = save_dir / 'tattoo_detection_results.csv'
    with_csv_path = save_dir / 'images_with_tattoos.csv'
    without_csv_path = save_dir / 'images_without_tattoos.csv'

    try:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image Name', 'Prediction', 'Confidence'])

        with_csv_file = open(with_csv_path, 'w', newline='', encoding='utf-8')
        with_csv_writer = csv.writer(with_csv_file)
        with_csv_writer.writerow(['Image Name'])

        without_csv_file = open(without_csv_path, 'w', newline='', encoding='utf-8')
        without_csv_writer = csv.writer(without_csv_file)
        without_csv_writer.writerow(['Image Name'])
    except IOError as e:
        LOGGER.error(f"Failed to open CSV files for writing in {save_dir}: {e}")
        raise

    # --- Model and Device Setup ---
    try:
        device = select_device(device)
        # Pass resolved string paths
        model = DetectMultiBackend(str(weights_path), device=device, dnn=False, data=data_path_str, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride) # Ensure imgsz is tuple e.g., (640, 640)

        # Get tattoo class index (handle potential case variations)
        tattoo_classid = None
        if isinstance(names, (dict, list)): # Handle both dict and list formats
            names_lower = {k: str(v).lower() for k, v in names.items()} if isinstance(names, dict) else [str(n).lower() for n in names]
            if isinstance(names_lower, dict):
                 for k, v_lower in names_lower.items():
                      if v_lower == 'tattoo':
                          tattoo_classid = k
                          break
            else: # List
                 try:
                     tattoo_classid = names_lower.index('tattoo')
                 except ValueError:
                      pass # Not found

        if tattoo_classid is None:
            LOGGER.warning(f"{colorstr('yellow', 'WARNING ⚠️')} 'tattoo' class not found in model classes: {names}. Tattoo detection might not work correctly.")
        else:
            LOGGER.info(f"Found 'tattoo' class with index: {tattoo_classid}")

    except Exception as e:
        LOGGER.error(f"Failed during model loading or setup: {e}")
        csv_file.close()
        with_csv_file.close()
        without_csv_file.close()
        raise # Re-raise the exception

    # --- Dataloader Setup (Images Only) ---
    try:
        # Pass resolved source path string
        dataset = LoadImages(str(source_path.resolve()), img_size=imgsz, stride=stride, auto=pt)
    except Exception as e:
         LOGGER.error(f"Failed to create image dataloader for source '{source_path.resolve()}': {e}")
         csv_file.close()
         with_csv_file.close()
         without_csv_file.close()
         raise

    # --- Run Inference ---
    try:
        model.warmup(imgsz=(1, 3, *imgsz)) # Warmup
        seen, dt = 0, (Profile(), Profile(), Profile())

        for path, im, im0s, _, s in dataset: # Iterate through images
            with dt[0]: # Preprocessing
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()
                im /= 255.0
                if len(im.shape) == 3:
                    im = im[None] # Expand for batch dim

            with dt[1]: # Inference
                pred = model(im, augment=False, visualize=False)

            with dt[2]: # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=False, max_det=1000)

            # --- Process detections for the single image ---
            for i, det in enumerate(pred): # Should only be one detection result for image dataloader
                seen += 1
                p = Path(path) # Original image path
                im0 = im0s.copy() # Original image for saving/annotation
                annotated_save_path = save_dir / p.name # Path for potentially annotated image

                s += '%gx%g ' % im.shape[2:] # Log image size
                annotator = Annotator(im0, line_width=3, example=str(names))

                # Tattoo Detection Logic
                found_tattoo = False
                highest_conf = 0.0
                best_tattoo_info = None

                if len(det):
                    # Rescale boxes from imgsz to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Check detections
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        if c == tattoo_classid: # Check if the detected class is 'tattoo'
                            found_tattoo = True
                            conf_item = conf.item()
                            if conf_item > highest_conf: # Track the highest confidence tattoo
                                highest_conf = conf_item
                                label = f'{names[c]} {highest_conf:.2f}'
                                best_tattoo_info = {
                                    "xyxy": xyxy,
                                    "label": label,
                                    "color": colors(c, True)
                                }

                # --- Write results to CSVs ---
                pred_value = found_tattoo
                conf_value = f"{highest_conf:.4f}" if found_tattoo else "0.0000"
                try:
                    csv_writer.writerow([p.name, pred_value, conf_value])
                    if found_tattoo:
                        with_csv_writer.writerow([p.name])
                    else:
                        without_csv_writer.writerow([p.name])
                except Exception as e:
                    LOGGER.error(f"Error writing row for {p.name} to CSV: {e}")


                # --- Copy Image Files (if flags are set) ---
                original_image_path = p.resolve() # Ensure we have the absolute path to copy from
                if found_tattoo:
                    pass
                    # if create_dataset_with_tattoo and with_tattoo_dir:
                    #     dest_path = with_tattoo_dir / p.name
                    #     try:
                    #         if original_image_path.is_file(): # Double check it's a file
                    #             shutil.copy2(original_image_path, dest_path) # copy2 preserves metadata
                    #     except Exception as e:
                    #         LOGGER.warning(f"Could not copy {original_image_path} to {dest_path}: {e}")
                else:
                    if create_dataset_without_tattoo and without_tattoo_dir:
                        dest_path = without_tattoo_dir / p.name
                        try:
                            if original_image_path.is_file():
                                shutil.copy2(original_image_path, dest_path)
                        except Exception as e:
                            LOGGER.warning(f"Could not copy {original_image_path} to {dest_path}: {e}")


                # --- Annotate and Save Image (Optional) ---
                if found_tattoo:
                    im0_annotated = annotator.result() # Start with original or potentially annotated
                    if found_tattoo and best_tattoo_info:
                        # Draw only the highest confidence tattoo box
                        annotator.box_label(best_tattoo_info["xyxy"], best_tattoo_info["label"], color=best_tattoo_info["color"])
                        im0_annotated = annotator.result() # Get image with the box drawn

                    # Save the potentially annotated image
                    if save_img:
                        try:
                             cv2.imwrite(str(annotated_save_path), im0_annotated)
                        except Exception as e:
                             LOGGER.error(f"Error saving annotated image {annotated_save_path}: {e}")

                    if create_dataset_with_tattoo and with_tattoo_dir:
                        try:
                            cv2.imwrite(str(with_tattoo_dir / p.name), im0_annotated)
                        except Exception as e:
                            LOGGER.error(f"Error saving image with tattoo {with_tattoo_dir / p.name}: {e}")

                # Log progress for this image
                det_info = f"{len(det)} detections" if len(det) else "no detections"
                if found_tattoo:
                    det_info += f", Tattoo conf: {highest_conf:.2f}"
                LOGGER.info(f"{s}{det_info} - {dt[1].dt * 1E3:.1f}ms inference, {dt[2].dt * 1E3:.1f}ms NMS")


    finally: # Ensure files are closed even if errors occur during inference
        # --- Finalization ---
        csv_file.close()
        with_csv_file.close()
        without_csv_file.close()

    LOGGER.info(f"Main predictions saved to {csv_path.resolve()}")
    if create_dataset_with_tattoo:
        LOGGER.info(f"Filenames with tattoos saved to {with_csv_path.resolve()}")
    if create_dataset_without_tattoo:
        LOGGER.info(f"Filenames without tattoos saved to {without_csv_path.resolve()}")

    t = tuple(x.t / seen * 1E3 if seen else 0 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    log_save_info = []
    if save_img:
        log_save_info.append(f"annotated images to {colorstr('bold', save_dir.resolve())}")
    if create_dataset_with_tattoo:
        log_save_info.append(f"images with tattoos copied to {colorstr('bold', with_tattoo_dir.resolve())}")
    if create_dataset_without_tattoo:
         log_save_info.append(f"images without tattoos copied to {colorstr('bold', without_tattoo_dir.resolve())}")

    if log_save_info:
         LOGGER.info(f"Results saved: {', '.join(log_save_info)}")
    else:
         LOGGER.info(f"Results logged to CSVs in {colorstr('bold', save_dir.resolve())}")

    return save_dir.resolve() # Return the absolute path to the results directory

# --- Optional: Keep command-line execution for direct testing ---
def parse_opt_for_testing():
    parser = argparse.ArgumentParser(description="Detect tattoos in images using YOLOv5.")
    # Arguments mirroring the function parameters, using defaults relative to YOLOV5_ROOT where appropriate
    parser.add_argument('--weights', type=str, default=str(YOLOV5_ROOT / 'best.pt'), help='model path')
    parser.add_argument('--source', type=str, required=True, help='path to image file or directory')
    parser.add_argument('--project', type=str, default='runs/detect_tattoos', help='save results relative to CWD')
    parser.add_argument('--name', type=str, default='exp', help='save results to project/name')
    parser.add_argument('--data', type=str, default=None, help=f'(optional) dataset.yaml path, relative to {YOLOV5_ROOT} if not absolute')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', action='store_true', help='do not save annotated images')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--create-dataset-with-tattoo', action='store_true', help='Create folder and copy images containing tattoos')
    parser.add_argument('--create-dataset-without-tattoo', action='store_true', help='Create folder and copy images not containing tattoos')

    opt = parser.parse_args()

    # Convert list from argparse back to tuple for imgsz
    imgsz_list = opt.imgsz * 2 if len(opt.imgsz) == 1 else opt.imgsz[:2]
    opt.imgsz = tuple(imgsz_list) # Ensure it's a tuple

    # Optional: Print arguments if running directly
    # from utils.general import print_args # Already imported
    # print_args(vars(opt))

    return opt

if __name__ == '__main__':
    # This block allows running this script directly for testing
    # Example: python yolov5_master/tattoo_detector.py --source ../my_images --weights yolov5s.pt --create-dataset-with-tattoo
    print(f"Running {Path(__file__).name} directly for testing...")
    try:
        opts = parse_opt_for_testing()
        detect_tattoos_in_images(**vars(opts)) # Call the main function
    except Exception as e:
        LOGGER.error(f"Error during direct execution: {e}", exc_info=True)
        sys.exit(1)
    print("Direct execution finished.")