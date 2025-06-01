import os
import shutil
from pathlib import Path


# helper function to increment path
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

def copy_diff_images(folder_original, folder_results, output_folder):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    files_a = set(os.listdir(folder_original))
    files_b = set(os.listdir(folder_results))

    # Find files in folder_original, but not in folder_results
    diff_files = files_a - files_b

    print(f"Copying {len(diff_files)} files to {output_folder} ")

    for filename in diff_files:
        src_path = os.path.join(folder_original, filename)
        dst_path = os.path.join(output_folder, filename)
        shutil.copy2(src_path, dst_path)

if __name__ == "__main__":

    # Todo: Set the path to the original folder
    folder_original = "test"

    # Todo: Set the path to the filtered_images folder
    folder_results = "output/CLIP_filter/filtered_images19"

    Results_dir = "output/CLIP_exclude"
    Name = "excluded_images"
    OUTPUT_DIR = increment_path(Results_dir, name=Name)

    copy_diff_images(folder_original, folder_results, OUTPUT_DIR)
