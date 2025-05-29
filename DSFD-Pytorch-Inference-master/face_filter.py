import glob
import os
import cv2
import time
import face_detection
import csv

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

if __name__ == "__main__":
    image_dir = "../data/original/D"  #"../images"  # Directory of images to process
    results_dir = "results"  # Directory to save results
    create_dataset_without_face = False  # Set to True if to create a dataset of images without faces
    create_dataset_with_face = True  # Set to True if to create a dataset of images with faces

    faces_dir = os.path.join(results_dir, "faces")
    no_faces_dir = os.path.join(results_dir, "no_faces")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.jfif", "*.bmp", "*.webp"]
    impaths = []

    for ext in image_extensions:
        impaths.extend(glob.glob(os.path.join(image_dir, ext)))

    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )

    main_csv_filename = os.path.join(results_dir, "face_detection_results.csv")
    faces_csv_filename = os.path.join(results_dir, "images_with_faces.csv")
    no_faces_csv_filename = os.path.join(results_dir, "images_without_faces.csv")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)
    os.makedirs(no_faces_dir, exist_ok=True)

    results = []
    imgs_with_faces = []
    imgs_without_faces = []
    start_time = time.time()

    print('impaths',len(impaths))
    for impath in impaths:
        im = cv2.imread(impath)
        filename = os.path.basename(impath)
        dest_path = os.path.join(results_dir, filename)  # Default destination

        if im is None:
            print(f"Error: Could not read image {impath}")
            results.append([filename, "Error: Could not read image"])
            continue

        print("Processing:", impath)
        t = time.time()

        try:
            dets = detector.detect(im[:, :, ::-1])[:, :4]
            draw_faces(im, dets)
        except Exception as e:
            print(f"Error during face detection on {impath}: {e}")
            results.append([filename, f"Error during detection: {e}"])
            # cv2.imwrite(dest_path, im)  # Save even if detection failed
            continue

        detection_time = time.time() - t
        num_faces = len(dets)
        faces_found = num_faces > 0

        results.append([filename, "True" if faces_found else "False"])

        if faces_found:
            print(f"Face(s) found in {impath}. Moving to '{faces_dir}'.")
            imgs_with_faces.append([filename])
            dest_path = os.path.join(faces_dir, filename)
            if create_dataset_with_face:
                cv2.imwrite(dest_path, im)
        else:
            print(f"No face found in {impath}. Moving to '{no_faces_dir}'.")
            imgs_without_faces.append([filename])
            dest_path = os.path.join(no_faces_dir, filename)
            if create_dataset_without_face:
                cv2.imwrite(dest_path, im)

    end_time = time.time()
    total_processing_time = end_time - start_time
    print(f"\nTotal processing time: {total_processing_time:.2f} seconds")

    # Write main results to CSV file
    with open(main_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["filename", "prediction (face detected)"]
        writer.writerow(header)
        writer.writerows(results)
    print(f"\nMain results saved to {main_csv_filename}")

    # Write list of images with faces to CSV file
    with open(faces_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["filename"]
        writer.writerow(header)
        writer.writerows(imgs_with_faces)
    print(f"List of images with faces saved to {faces_csv_filename}")

    # Write list of images without faces to CSV file
    with open(no_faces_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["filename"]
        writer.writerow(header)
        writer.writerows(imgs_without_faces)
    print(f"List of images without faces saved to {no_faces_csv_filename}")

    # Print the pecentage of images with and with faces
    total_images = len(impaths)
    images_with_faces = len(imgs_with_faces)
    percentage_with_faces = (images_with_faces / total_images) * 100
    print(f"\nPercentage of images with faces: {percentage_with_faces:.2f}%")

    print("Processing complete.")
