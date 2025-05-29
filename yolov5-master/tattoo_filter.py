import sys
import pathlib
if sys.platform != "win32":
    # 如果不是 Windows，就让 WindowsPath 指向 PosixPath
    pathlib.WindowsPath = pathlib.PosixPath

import argparse
import csv
import os
from pathlib import Path
import shutil # <--- Added for file copying
import torch

# Ensure ROOT is correctly defined relative to the script's location
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # Assuming the script is in the yolov5 root or similar
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Make ROOT relative to CWD for paths in args if needed, but absolute paths are often safer
# ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Commenting out, might cause issues depending on execution context

# --- Check if these imports are still correct for your yolov5 version ---
try:
    from ultralytics.utils.plotting import Annotator, colors
    # Assuming models and utils are relative to ROOT or in sys.path
    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadScreenshots
    from utils.general import (
        LOGGER,
        Profile,
        check_file,
        check_img_size,
        check_imshow,
        check_requirements,
        colorstr,
        cv2,
        increment_path,
        non_max_suppression,
        print_args,
        scale_boxes,
        # xyxy2xywh, # No longer needed for this specific output format
    )
    from utils.torch_utils import select_device, smart_inference_mode
except ImportError as e:
    LOGGER.error(f"ImportError: {e}. Make sure your YOLOv5 environment is set up correctly and dependencies are installed.")
    LOGGER.error(f"Current sys.path: {sys.path}")
    LOGGER.error(f"Expected ROOT: {ROOT}")
    sys.exit(1)
# --- End Import Check ---

@smart_inference_mode()
def run(
    weights=ROOT / 'best.pt',        # Default path relative to script location
    source='../images',  # ROOT / 'data/images',  # Default path relative to script location
    data=ROOT / 'data/coco128.yaml', # Default path relative to script location
    imgsz=(640, 640),                # inference size (height, width)
    conf_thres=0.25,                 # confidence threshold
    iou_thres=0.45,                  # NMS IOU threshold
    device='',                       # cuda device, i.e. 0 or 0,1,2,3 or cpu
    # save_csv=True,                 # Keep main CSV saving enabled implicitly
    nosave=False,                    # do not save annotated images/videos
    exist_ok=False,                  # existing project/name ok, do not increment
    project=ROOT / 'runs/detect',    # save results to project/name
    name='exp',                      # save results to project/name
    classes=None,                    # filter by class: --class 0, or --class 0 2 3
    create_dataset_with_tattoo=False,  # <--- New flag: If True, copy images with tattoos
    create_dataset_without_tattoo=False, # <--- New flag: If True, copy images without tattoos
    # ... (other potential yolov5 args)
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    is_screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    save_img = not nosave  # save annotated inference images
    # save_img = False

    # --- Create dataset directories if flags are set ---
    with_tattoo_dir = None
    without_tattoo_dir = None
    if create_dataset_with_tattoo:
        with_tattoo_dir = save_dir / 'images_with_tattoo'
        with_tattoo_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Will copy images with tattoos to: {with_tattoo_dir}")
    if create_dataset_without_tattoo:
        without_tattoo_dir = save_dir / 'images_without_tattoo'
        without_tattoo_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Will copy images without tattoos to: {without_tattoo_dir}")
    # --- End dataset directory creation ---

    # --- CSV Setup ---
    # Main predictions CSV
    csv_path = save_dir / 'tattoo_detection_results.csv'
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image Name', 'Prediction', 'Confidence'])

    # CSV for filenames with tattoos
    with_csv_path = save_dir / 'images_with_tattoos.csv'
    with_csv_file = open(with_csv_path, 'w', newline='', encoding='utf-8')
    with_csv_writer = csv.writer(with_csv_file)
    with_csv_writer.writerow(['Image Name'])

    # CSV for filenames without tattoos
    without_csv_path = save_dir / 'images_without_tattoos.csv'
    without_csv_file = open(without_csv_path, 'w', newline='', encoding='utf-8')
    without_csv_writer = csv.writer(without_csv_file)
    without_csv_writer.writerow(['Image Name'])
    # --- End CSV Setup ---


    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Get tattoo class index
    tattoo_classid = None
    if isinstance(names, dict): # Standard YOLOv5 format
        for k, v in names.items():
            if v.lower() == 'tattoo':
                tattoo_classid = k
                break
    elif isinstance(names, list): # Some Ultralytics formats might use lists
        try:
            tattoo_classid = names.index('tattoo') # Case-sensitive
        except ValueError:
            # Try case-insensitive
             for i, name in enumerate(names):
                 if name.lower() == 'tattoo':
                     tattoo_classid = i
                     break

    if tattoo_classid is None:
        LOGGER.warning(f"{colorstr('yellow', 'WARNING ⚠️')} 'tattoo' class not found in model classes: {names}. All images will be classified as 'False' / 'without_tattoo'.")
    else:
         LOGGER.info(f"Found 'tattoo' class with index: {tattoo_classid}")


    # Dataloader
    bs = 1
    if is_webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        bs = len(dataset)
    elif is_screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=False, max_det=1000)

        # Process predictions per image
        for i, det in enumerate(pred):
            seen += 1
            if is_webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # Source image path
            annotated_save_path = str(save_dir / p.name) # Path for potentially annotated image
            s += '%gx%g ' % im.shape[2:]
            annotator = Annotator(im0.copy(), line_width=3, example=str(names)) # Use a copy for annotation

            # --- Tattoo Detection Logic ---
            found_tattoo = False
            highest_conf = 0.0
            best_tattoo_info = None

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    if c == tattoo_classid:
                        found_tattoo = True
                        conf_item = conf.item()
                        if conf_item > highest_conf:
                            highest_conf = conf_item
                            label = f'{names[c]} {highest_conf:.2f}'
                            best_tattoo_info = {
                                "xyxy": xyxy,
                                "label": label,
                                "color": colors(c, True)
                            }

            # --- Write to CSVs ---
            pred_value = found_tattoo
            conf_value = f"{highest_conf:.4f}" if found_tattoo else "0.0000"
            try:
                # Main CSV
                csv_writer.writerow([p.name, pred_value, conf_value])
                # Conditional CSVs
                if found_tattoo:
                    with_csv_writer.writerow([p.name])
                else:
                    without_csv_writer.writerow([p.name])
            except Exception as e:
                 LOGGER.error(f"Error writing row for {p.name} to CSV: {e}")
            # --- End CSV Writing ---

            # --- Copy Image Files (if flags are set) ---
            if found_tattoo:
                if create_dataset_with_tattoo and with_tattoo_dir:
                    dest_path = with_tattoo_dir / p.name
                    try:
                        shutil.copy2(p, dest_path) # copy2 preserves metadata
                    except Exception as e:
                         LOGGER.warning(f"Could not copy {p} to {dest_path}: {e}")
            else:
                if create_dataset_without_tattoo and without_tattoo_dir:
                    dest_path = without_tattoo_dir / p.name
                    try:
                        shutil.copy2(p, dest_path) # copy2 preserves metadata
                    except Exception as e:
                         LOGGER.warning(f"Could not copy {p} to {dest_path}: {e}")
            # --- End Image Copying ---


            # --- Annotate and Save Image (Optional) ---
            im0_annotated = annotator.result() # Get original image back if no annotation happened
            if found_tattoo and best_tattoo_info and save_img:
                # Draw only the box with the highest confidence
                annotator.box_label(best_tattoo_info["xyxy"], best_tattoo_info["label"], color=best_tattoo_info["color"])
                im0_annotated = annotator.result() # Get image with the single box drawn

            if found_tattoo and save_img:
                if dataset.mode == 'image':
                    try:
                        cv2.imwrite(annotated_save_path, im0_annotated)
                    except Exception as e:
                        LOGGER.error(f"Error saving annotated image {annotated_save_path}: {e}")
                else:  # 'video' or 'stream'
                    if vid_path[i] != annotated_save_path:
                        vid_path[i] = annotated_save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0_annotated.shape[1], im0_annotated.shape[0]
                        annotated_save_path = str(Path(annotated_save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(annotated_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0_annotated)
                    except Exception as e:
                         LOGGER.error(f"Error writing video frame to {annotated_save_path}: {e}")
            # --- End Annotate and Save ---

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


    # --- Finalization ---
    # Close all CSV files
    csv_file.close()
    with_csv_file.close()
    without_csv_file.close()

    LOGGER.info(f"Main predictions saved to {str(csv_path.resolve())}")
    LOGGER.info(f"Filenames with tattoos saved to {str(with_csv_path.resolve())}")
    LOGGER.info(f"Filenames without tattoos saved to {str(without_csv_path.resolve())}")

    t = tuple(x.t / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    log_save_info = []
    if save_img:
        log_save_info.append(f"annotated images/videos to {colorstr('bold', save_dir)}")
    if create_dataset_with_tattoo:
        log_save_info.append(f"images with tattoos copied to {colorstr('bold', with_tattoo_dir)}")
    if create_dataset_without_tattoo:
         log_save_info.append(f"images without tattoos copied to {colorstr('bold', without_tattoo_dir)}")

    if log_save_info:
         LOGGER.info(f"Results saved: {', '.join(log_save_info)}")
    # --- End Finalization ---


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/woody/iwso/iwso183h/yolov5-master/best.pt', help='model path(s)')  # default=ROOT / 'best.pt'
    parser.add_argument('--source', type=str, default='/home/woody/iwso/iwso183h/data/original/D', help='file/dir/URL/glob/screen/0(webcam)')  # default=ROOT / 'data/images'
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')  # 0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', action='store_true', help='do not save annotated images/videos')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # --- New Arguments ---
    parser.add_argument('--create-dataset-with-tattoo', action='store_true', help='Create folder and copy images containing tattoos')
    parser.add_argument('--create-dataset-without-tattoo', action='store_true', help='Create folder and copy images not containing tattoos')
    # --- End New Arguments ---

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    # Print arguments, including the new ones
    print_args(vars(opt))
    return opt


def main(opt):
    # Optional: Uncomment if requirements check is needed frequently
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


