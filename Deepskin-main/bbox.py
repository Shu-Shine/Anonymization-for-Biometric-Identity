
import cv2
import numpy as np
import pylab as plt
from deepskin import wound_segmentation
# evaluate_PWAT_score
from deepskin import evaluate_PWAT_score

def crop_and_analyze_wound(img_path, output_filename="output_cropped.png", margin=50):
    """
    Loads an image, performs wound segmentation, crops the image and masks
    around the detected wound with a specified margin, and saves a visualization.

    Args:
        img_path (str): Path to the input image file.
        output_filename (str): Path to save the output visualization PNG.
        margin (int): The number of pixels to add as padding around the
                      wound's bounding box for cropping.
    """
    print(f"Processing image: {img_path}")
    print(f"Cropping margin: {margin} pixels")

    # Load and prepare the image
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Error: Could not load image at {img_path}")
        return
    rgb = bgr[..., ::-1] # Convert BGR to RGB
    h, w = rgb.shape[:2]
    print(f"Original image size: {w}x{h}")

    # Perform wound segmentation
    try:
        segmentation = wound_segmentation(img=rgb)
        wound_mask, body_mask, bg_mask = cv2.split(segmentation)
    except Exception as e:
        print(f"Error during wound segmentation: {e}")
        return

    # Find bounding box of the wound
    # Find coordinates of all non-zero pixels (wound pixels)
    coords = np.argwhere(wound_mask > 0)

    if coords.size == 0:
        print("Warning: No wound detected in the segmentation mask. Skipping cropping.")
        # display the original image and full segmentation
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        # ax1.imshow(rgb)
        # ax1.set_title('Original Image (No Wound Detected)', fontsize=18)
        # _ = ax1.axis('off')
        # ax2.imshow(segmentation)
        # ax2.set_title('Segmentation mask', fontsize=18)
        # _ = ax2.axis('off')
        # plt.tight_layout()
        # plt.savefig(output_filename)
        # plt.close(fig) # Close the figure to free memory
        # print(f"Saved visualization without cropping to {output_filename}")

        plt.figure(figsize=(7, 7))
        plt.imshow(rgb)
        plt.title('Original Image (No Wound Detected)', fontsize=20)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()  # No need to pass a figure here
        print(f"Saved visualization without cropping to {output_filename}")

        return

    # Determine min/max row (y) and column (x)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    print(f"Wound bounding box (y1, x1, y2, x2): ({min_y}, {min_x}, {max_y}, {max_x})")

    # Calculate crop coordinates with margin, clamping to image boundaries
    crop_x1 = max(0, min_x - margin)
    crop_y1 = max(0, min_y - margin)
    crop_x2 = min(w, max_x + margin + 1) # +1 because slicing is exclusive at the end
    crop_y2 = min(h, max_y + margin + 1) # +1 because slicing is exclusive at the end
    print(f"Crop coordinates with margin (y1:y2, x1:x2): ({crop_y1}:{crop_y2}, {crop_x1}:{crop_x2})")

    # Crop the images and masks
    cropped_rgb = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_segmentation = segmentation[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_wound_mask = wound_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_body_mask = body_mask[crop_y1:crop_y2, crop_x1:crop_x2]

    # --- Compute PWAT score (usually done on original image) ---
    # try:
    #     pwat = evaluate_PWAT_score(img=rgb, mask=segmentation)
    #     print(f'PWAT score: {pwat:.3f}')
    # except Exception as e:
    #     print(f"Could not compute PWAT score: {e}")
    #     pwat = None
    # --------------------------------------------------------------------

    # Display the cropped results
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
    #
    # ax1.imshow(cropped_rgb)
    # ax1.set_title('Cropped Original', fontsize=20)
    # _ = ax1.axis('off')

    # ax2.imshow(cropped_segmentation)
    # ax2.set_title('Cropped Segmentation Mask', fontsize=20)
    # _ = ax2.axis('off')
    #
    # ax3.imshow(cropped_rgb)
    # # Adjust contour levels if needed, usually 0.5 for binary masks
    # ax3.contour(cropped_body_mask, levels=[0.5], colors='blue', linewidths=1)
    # ax3.contour(cropped_wound_mask, levels=[0.5], colors='lime', linewidths=2)
    # ax3.set_title('Cropped Semantic Segmentation', fontsize=20)
    # _ = ax3.axis('off')

    # plt.tight_layout() # Adjust layout to prevent overlap
    # plt.savefig(output_filename)
    # plt.close(fig) # Close the figure to free memory
    # print(f"Saved cropped visualization to {output_filename}")

    plt.figure(figsize=(7, 7))
    plt.imshow(cropped_rgb)
    plt.title('Cropped Original', fontsize=20)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved cropped visualization to {output_filename}")



if __name__ == "__main__":
    # Define the input image path
    img_path = '../data/original/V/27.jpg'
    output_path = '../output/cropped/D/1.jpg'


    crop_and_analyze_wound(img_path, output_filename="../output/V_wound_cropped_margin_50.png", margin=50)

    crop_and_analyze_wound(img_path, output_filename="../output/V_wound_cropped_margin_20.png", margin=20)

    crop_and_analyze_wound(img_path, output_filename="../output/V_wound_cropped_margin_80.png", margin=80)
#
