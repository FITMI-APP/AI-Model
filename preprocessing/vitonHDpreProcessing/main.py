# This is a sample Python script.
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
output_img_densePose_path=base_path + "test/densepose"

def preprocessVitonhd():
    remove_background(input_img_path, output_img_BG_remove_path, False)
    parsingExtractor(output_img_BG_remove_path, output_img_parsing_path, False)
    openposeExtractor(output_img_BG_remove_path + "/", output_img_openPose_path + "/", output_img_Json_path + "/")
    denseposeExtractor(output_img_BG_remove_path,output_img_densePose_path)
    remove_background(input_cloth_path, output_cloth_mask_path, True)
    remove_background(input_cloth_path, output_cloth_BG_remove_path, False)
# preprocessVitonhd()