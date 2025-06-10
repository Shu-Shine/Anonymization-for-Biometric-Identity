import os
import pandas as pd
import shutil
from pathlib import Path

# ---Configuration---
# todo: Check and copy single type images into a subfolder
is_check = False  # True, False
check_class = "P"

# todo: Paths to Excels and image folder
patient_file = "output/wound_types.xlsx"
image_file = "output/case_images.xlsx"
image_folder = "output/CLIP_filter/filtered_images2"

# todo: Column names in the Excels
case_id = 'case_id'
image_name = 'image_name'
diagnosis = 'diagnosis'

# ---Utilities---
# def increment_path(base_dir, name='test'):
#     base = Path(base_dir)
#     path = base / name
#     if not path.exists():
#         return path.resolve()
#     else:
#         n = 2
#         while (base / f"{name}{n}").exists():
#             n += 1
#         return (base / f"{name}{n}").resolve()

Root_output = "output/Wound_type_counts"
output_dir = os.path.join(Root_output, "counts")
os.makedirs(output_dir, exist_ok=True)

# ---Execution Starts---
if not is_check:
    # output_dir = increment_path(Root_output, "counts")
    # os.makedirs(output_dir, exist_ok=True)

    df_patient = pd.read_excel(patient_file)
    df_images = pd.read_excel(image_file)
    df = pd.merge(df_images, df_patient, on=case_id, how='inner')
    existing_images = set(os.listdir(image_folder))
    df = df[df[image_name].isin(existing_images)]
    grouped = df.groupby(diagnosis)

    # Save count summary
    count_per_type = grouped.size().to_dict()
    print("Image counts per wound type:")
    for wound_type, count in count_per_type.items():
        print(f"{wound_type}: {count}")

    count_file = os.path.join(output_dir, "wound_type_counts.txt")
    if not os.path.exists(count_file):
        with open(count_file, 'w') as f:
            for wound_type, count in count_per_type.items():
                f.write(f"{wound_type}: {count}\n")

    for wound_type, group in grouped:
        txt_path = os.path.join(output_dir, f"{wound_type}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, 'w') as f:
                for img in group[image_name].tolist():
                    f.write(f"{img}\n")
else:
    print("[INFO] Skipping data preparation, using existing txts.")


# ---Class Copy Section---
class_path = os.path.join(output_dir, f"{check_class}.txt")

if not os.path.exists(class_path):
    print(f"[ERROR] Class selection file not found: {check_class}")
    check_class = []

for wound_type in check_class:
    txt_path = os.path.join(output_dir, f"{wound_type}.txt")
    if not os.path.exists(txt_path):
        print(f"[WARNING] Image list not found for '{wound_type}': {txt_path}")
        continue

    class_folder = os.path.join(Root_output, wound_type)
    os.makedirs(class_folder, exist_ok=True)

    with open(txt_path, 'r') as f:
        image_list = [line.strip() for line in f if line.strip()]

    for img_name in image_list:
        src_path = os.path.join(image_folder, img_name)
        dst_path = os.path.join(class_folder, img_name)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

