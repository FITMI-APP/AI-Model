import glob
import os
from pathlib import Path

from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

PARENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

def load_model_and_data():
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = tensorflow.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    model.trainable = False
    # Construct paths relative to the directory of main.py
    embeddings_path = PARENT_ROOT / 'embeddings.pkl'
    filenames_path = PARENT_ROOT / 'filenames.pkl'

    # Load pre-computed features and filenames
    feature_list = np.array(pickle.load(open(embeddings_path, 'rb')))
    filenames = pickle.load(open(filenames_path, 'rb'))
    print(filenames)

    return model, feature_list, filenames

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list, filenames):
    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    recommended_images = [filenames[idx] for idx in indices[0]]
    return recommended_images
def male_complementary_recommend_fashion_item(folder_path):
    # Load model and data
    model, feature_list, filenames = load_model_and_data()

    # Initialize a list to store recommended images
    all_recommended_images = []

    # Iterate over all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Construct the full path to the image
            img_path = os.path.join(folder_path, filename)

            # Feature extraction
            features = feature_extraction(img_path, model)

            # Recommendation
            recommended_images = recommend(features, feature_list, filenames)

            # Append recommended images to the list
            all_recommended_images.append(recommended_images)

            # Save recommended images to the output folder
            # Remove existing files in the output folder
            output_folder = PARENT_ROOT / "output"
            existing_files = glob.glob(os.path.join(output_folder, "*"))
            for existing_file in existing_files:
                os.remove(existing_file)

            os.makedirs(output_folder, exist_ok=True)
            for i, img_path in enumerate(recommended_images):
                img = Image.open(Path(__file__).resolve().parent / img_path)
                img.save(os.path.join(output_folder, f'{filename}_recommended_image_{i}.jpg'))

    return all_recommended_images

# # Example usage:
# cloth_folder_path = "E:\FCAI - HU/LEVEL 4/GP/GP Dataset/viton-hd/test/test_Dress_code/upper_body/cloth_BG"
# recommended_images_list = recommend_fashion_item(cloth_folder_path)
# print("Recommended Images:")
# for recommended_images in recommended_images_list:
#     for img_path in recommended_images:
#         print(img_path)