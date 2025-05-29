import os
import cv2
import numpy as np
import pylab as plt
from deepskin import wound_segmentation


def crop_and_analyze_wound(img_path, output_filename="output_cropped.png", margin=50):
    print(f"Processing image: {img_path}")
    print(f"Cropping margin: {margin} pixels")

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Error: Could not load image at {img_path}")
        return
    rgb = bgr[..., ::-1]
    h, w = rgb.shape[:2]

    try:
        segmentation = wound_segmentation(img=rgb)
        wound_mask, body_mask, bg_mask = cv2.split(segmentation)
    except Exception as e:
        print(f"Error during wound segmentation: {e}")
        return

    coords = np.argwhere(wound_mask > 0)
    if coords.size == 0:
        print("Warning: No wound detected in the segmentation mask. Skipping cropping.")

        # plt.figure(figsize=(7, 7))
        # plt.imshow(rgb)
        # plt.title('Original Image (No Wound Detected)', fontsize=20)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(output_filename)
        # plt.close()
        # print(f"Saved visualization without cropping to {output_filename}")
        return

    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    crop_x1 = max(0, min_x - margin)
    crop_y1 = max(0, min_y - margin)
    crop_x2 = min(w, max_x + margin + 1)
    crop_y2 = min(h, max_y + margin + 1)

    cropped_rgb = rgb[crop_y1:crop_y2, crop_x1:crop_x2]

    plt.figure(figsize=(7, 7))
    plt.imshow(cropped_rgb)
    # plt.title('Cropped Original', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved cropped visualization to {output_filename}")


def batch_crop_wound(input_dir, output_dir, margins=[50]):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        name, ext = os.path.splitext(filename)

        for margin in margins:
            # out_name = f"{name}_margin{margin}{ext}"
            out_name = f"{name}.png"
            out_path = os.path.join(output_dir, out_name)
            crop_and_analyze_wound(img_path, output_filename=out_path, margin=margin)


if __name__ == "__main__":
    input_folder = "../output/CLIP_filter"
    output_folder = "../output/Crop/50"
    # margins = [20, 50, 80]
    margins = [50]

    batch_crop_wound(input_folder, output_folder, margins)
