import os
import pandas as pd
from pathlib import Path

# todo: pip install pandas openpyxl

# ---Configuration---
# Paths to files and folders
patient_file = "output/wound_types.xlsx"   # todo
image_file = "output/case_images.xlsx"   # todo
image_folder = "output/CLIP_filter/filtered_images2"  # todo

# Column names
case_id = 'case_id'   # todo
image_name = 'image_name'  # todo
diagnosis = 'diagnosis'  # todo


# ---Main Script---
def increment_path(base_dir, name='test'):
    base = Path(base_dir)
    path = base / name
    if not path.exists():
        return path.resolve()
    else:
        n = 2
        while (base / f"{name}{n}").exists():
            n += 1
        return (base / f"{name}{n}").resolve()

# Load Excels
df_patient = pd.read_excel(patient_file)       # for case_id, wound_type
df_images = pd.read_excel(image_file)       # for case_id, image_name

# Merge on 'case_id' to associate image_name with wound_type
df = pd.merge(df_images, df_patient, on=case_id, how='inner')

# Filter images that exist in the image folder
existing_images = set(os.listdir(image_folder))
df = df[df[image_name].isin(existing_images)]

# Group by wound_type
grouped = df.groupby('diagnosis')

# Count images per wound type
count_per_type = grouped.size().to_dict()
print("Image counts per wound type:")
for wound_type, count in count_per_type.items():
    print(f"{wound_type}: {count}")


output_dir = increment_path("output/Wound_type_counts", "counts")
os.makedirs(output_dir, exist_ok=True)

# Save the counts to a text file
with open(os.path.join(output_dir, "wound_type_counts.txt"), 'w') as f:
    for wound_type, count in count_per_type.items():
        f.write(f"{wound_type}: {count}\n")

# Save image names for each wound type
for wound_type, group in grouped:
    images = group[image_name].tolist()
    with open(os.path.join(output_dir, f"{wound_type}_images.txt"), 'w') as f:
        for img in images:
            f.write(f"{img}\n")
