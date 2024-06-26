from pathlib import Path
import cv2
import numpy as np
import glob
import json
import os

from preprocessing.vitonHDpreProcessing.OpenPose.src import util
from preprocessing.vitonHDpreProcessing.OpenPose.src.body import Body

PARENT_ROOT = Path(__file__).resolve().parent
body_estimation = Body(PARENT_ROOT / "model" / "body_pose_model.pth")



def openposeExtractor(input_path,output_path,keypoint_path):
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
  for image_path in glob.glob(input_path + '*'):
    image_filename = os.path.basename(image_path)
    oriImg = cv2.imread(image_path)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(np.zeros_like(oriImg), candidate, subset)
    arr = candidate.tolist()
    vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    for i in range(0, 18):
      if len(arr) == i or arr[i][3] != vals[i]:
        arr.insert(i, [-1, -1, -1, vals[i]])

    keypoints = {'keypoints': arr[:18]}
    cv2.imwrite(output_path + image_filename, canvas)
    print(image_path+"openpose image saved")
    with open(keypoint_path + os.path.splitext(image_filename)[0] + ".json", 'w') as fin:
      fin.write(json.dumps(keypoints))
    print(image_path+"openpose json saved")


