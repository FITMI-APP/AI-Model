# This is a sample Python script.
import os
from pathlib import Path
from preprocessing.vitonHDpreProcessing.Mask.mask import remove_background
from preprocessing.vitonHDpreProcessing.OpenPose.openpose import openposeExtractor
from preprocessing.vitonHDpreProcessing.Parsing.simple_extractor import parsingExtractor
from preprocessing.vitonHDpreProcessing.DensePose.projects.DensePose.apply_net import denseposeExtractor

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
base_path = str(PROJECT_ROOT / "datasets" / "vitonHDDataset") + "/"

input_img_path = base_path + "test/image"
output_img_parsing_path = base_path + "test/image-parse-v3"
output_img_openPose_path = base_path + "test/openpose_img"
output_img_Json_path = base_path + "test/openpose_json"
output_img_BG_remove_path = base_path + "test/image_BG"
output_cloth_BG_remove_path = base_path + "test/cloth_BG"
input_cloth_path = base_path + "test/cloth"
output_cloth_mask_path = base_path + "test/cloth-mask"
output_img_densePose_path = base_path + "test/densepose"


def preprocessVitonhd():
    remove_background(input_img_path, output_img_BG_remove_path, False)
    parsingExtractor(output_img_BG_remove_path, output_img_parsing_path, False)
    openposeExtractor(output_img_BG_remove_path + "/", output_img_openPose_path + "/", output_img_Json_path + "/")
    denseposeExtractor(output_img_BG_remove_path, output_img_densePose_path)
    remove_background(input_cloth_path, output_cloth_mask_path, True)
    remove_background(input_cloth_path, output_cloth_BG_remove_path, False)


def VitonHD_replace_file_extension_with_jpg():
    image_files = os.listdir(input_img_path)
    cloth_files = os.listdir(input_cloth_path)

    # Iterate over files in the folder
    for filename in image_files:
        # Check if the file has a .png extension
        # print(root)
        # print(filename)
        if filename.endswith('.png'):
            # Generate the new filename with .jpg extension
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            # Construct the full path for both old and new filenames
            old_path = os.path.join(input_img_path, filename)
            new_path = os.path.join(input_img_path, new_filename)
            # Rename the file
            os.rename(old_path, new_path)
            print(f"image Renamed {filename} to {new_filename}")
    for filename in cloth_files:
        # Check if the file has a .png extension
        # print(root)
        # print(filename)
        if filename.endswith('.png'):
            # Generate the new filename with .jpg extension
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            # Construct the full path for both old and new filenames
            old_path = os.path.join(input_cloth_path, filename)
            new_path = os.path.join(input_cloth_path, new_filename)
            # Rename the file
            os.rename(old_path, new_path)
            print(f"cloth Renamed {filename} to {new_filename}")


def VitonHD_update_pairs_file():
    # List files in the directory
    person_files = os.listdir(input_img_path)
    cloth_files = os.listdir(input_cloth_path)

    pairs_file_path = base_path + "test_pairs.txt"
    with open(pairs_file_path, 'w') as pairs_file:  # Open file in write mode to clear existing content
        pairs_file.write(f"{person_files[-1]} {cloth_files[-1]}\n")
        print("pair file is updated")

# preprocessVitonhd()
# VitonHD_replace_file_extension_with_jpg()
#VitonHD_update_pairs_file()
