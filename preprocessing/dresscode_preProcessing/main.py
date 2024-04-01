import os
from pathlib import Path

from preprocessing.dresscode_preProcessing.Mask.mask import remove_background
from preprocessing.dresscode_preProcessing.OpenPose.openpose import openposeExtractor
from preprocessing.dresscode_preProcessing.Parsing.simple_extractor import parsingExtractor
from preprocessing.dresscode_preProcessing.DensePose.projects.DensePose.apply_net import denseposeExtractor
PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
base_path = str(PROJECT_ROOT / "datasets" / "dresscodeDataset") + "/"


def preprocessDresscode(category):
    # List files in the directory
    root = base_path + category
    files = os.listdir(root + "/images")

    # Separate person images and cloth images
    person_images = [file for file in files if file.endswith('_0.jpg') or file.endswith("_0.jpeg") or file.endswith("_0.png")]
    cloth_images = [file for file in files if file.endswith('_1.jpg') or file.endswith("_1.jpeg") or file.endswith("_1.png")]
    print("All files in directory:", files)
    print("Person images:", person_images)
    print("Cloth images:", cloth_images)

    # Process person images
    for person_image in person_images:
        person_image_path = os.path.join(root + "/images", person_image)
        remove_background(person_image_path, root + "/image_BG", False)
    # Process cloth images
    for cloth_image in cloth_images:
        cloth_image_path = os.path.join(root + "/images", cloth_image)
        remove_background(cloth_image_path, root + "/masks", True)
        remove_background(cloth_image_path, root + "/cloth_BG", False)

    parsingExtractor(root + "/image_BG", root + "/label_maps", False)
    openposeExtractor(root + "/image_BG" + "/", root + "/skeletons" + "/", root + "/keypoints" + "/")
    denseposeExtractor(root + "/image_BG", root + "/dense")


# preProcess("upper_body")