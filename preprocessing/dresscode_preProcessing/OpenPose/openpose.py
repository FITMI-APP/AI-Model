from pathlib import Path

import cv2
import numpy as np
import glob
import json
import os

from preprocessing.dresscode_preProcessing.OpenPose.src import util
from preprocessing.dresscode_preProcessing.OpenPose.src.body import Body

PARENT_ROOT = Path(__file__).resolve().parent
body_estimation = Body(PARENT_ROOT / "model" / "body_pose_model.pth")


def openposeExtractor(input_path, output_path, keypoint_path):
    # Delete existing files in the output directory
    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file: {e}")
    if os.path.exists(keypoint_path):
        for file in os.listdir(keypoint_path):
            file_path = os.path.join(keypoint_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file: {e}")
    for image_path in glob.glob(input_path + '/*'):  # Add a separator at the end of input_path
        image_filename = os.path.basename(image_path)
        Img_out_name = image_filename.replace("_0.", "_5.")
        json_out_name = image_filename.replace("_0.", "_2.")
        out_image_name_only, image_ext = os.path.splitext(Img_out_name)
        out_json_name_only, image_ext = os.path.splitext(json_out_name)
        oriImg = cv2.imread(image_path)  # B,G,R order
        candidate, subset = body_estimation(oriImg)
        canvas = util.draw_bodypose(np.zeros_like(oriImg), candidate, subset)
        arr = candidate.tolist()
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        for i in range(0, 18):
            if len(arr) == i or arr[i][3] != vals[i]:
                arr.insert(i, [-1, -1, -1, vals[i]])

        keypoints = {'keypoints': arr[:18]}
        output_image_name = f"{out_image_name_only}.jpg"  # Output image filename
        output_json_name = f"{out_json_name_only}.json"  # Output JSON filename
        cv2.imwrite(os.path.join(output_path, output_image_name), canvas)  # Use os.path.join to create paths
        print(f"{image_path} openpose image saved as {output_image_name}")
        with open(os.path.join(keypoint_path, output_json_name), 'w') as fin:
            fin.write(json.dumps(keypoints))
        print(f"{image_path} openpose json saved as {output_json_name}")
