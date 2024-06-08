import shutil
from flask import Flask, request, jsonify, send_file, abort, Response, make_response
from pathlib import Path
import os
from PIL import Image
from sympy import true, false
from io import BytesIO
from PIL import Image


app = Flask(__name__)




PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
def update_paths_recom(category):

    global Cloth_Recommendation_FOLDER

    paths_dict = {
        "upper_body": {
            "Cloth_Recommendation_FOLDER": PROJECT_ROOT / "recommendationSystem" / "upper_body" / "output"
        },
        "lower_body": {
            "Cloth_Recommendation_FOLDER": PROJECT_ROOT / "recommendationSystem" / "lower_body" / "output"
        },
        "dresses": {
            "Cloth_Recommendation_FOLDER": PROJECT_ROOT / "recommendationSystem" / "dresses" / "output"
        }
    }

    try:
        paths = paths_dict[category]
        Cloth_Recommendation_FOLDER = paths["Cloth_Recommendation_FOLDER"]
    except KeyError:
        raise ValueError("Invalid category")
    

def update_paths_recom_comp(gender):
    global Cloth_Recommendation_FOLDER_Comp

    paths_dict = {
        "male": {
            "Cloth_Recommendation_FOLDER": PROJECT_ROOT / "recommendationSystem" / "complementary" / "male" / "output"
        },
        "female": {
            "Cloth_Recommendation_FOLDER": PROJECT_ROOT / "recommendationSystem" / "complementary" / "female" / "output"
        },
    }

    try:
        paths = paths_dict[gender]
        Cloth_Recommendation_FOLDER_Comp = paths["Cloth_Recommendation_FOLDER"]
    except KeyError:
        raise ValueError("Invalid category")


def update_paths(gender, category):
    global UPLOAD_FOLDER, CLOTHES_FOLDER, OUTPUT_FOLDER

    paths_dict = {
        ("male", "upper_body"): {
            "UPLOAD_FOLDER": PROJECT_ROOT / "datasets" / "vitonHDDataset" / "test" / "image",
            "CLOTHES_FOLDER": PROJECT_ROOT / "datasets" / "vitonHDDataset" / "test" / "cloth",
            "OUTPUT_FOLDER": PROJECT_ROOT / "datasets" / "vitonHDDataset" / "test" / "output" / "unpaired" / "upper_body"
        },
        ("male", "lower_body"): {
            "UPLOAD_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "lower_body" / "images",
            "CLOTHES_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "lower_body" / "images",
            "OUTPUT_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "lower_body" / "output" / "unpaired" / "lower_body"
        },
        ("female", "upper_body"): {
            "UPLOAD_FOLDER": PROJECT_ROOT / "datasets" / "vitonHDDataset" / "test" / "image",
            "CLOTHES_FOLDER": PROJECT_ROOT / "datasets" / "vitonHDDataset" / "test" / "cloth",
            "OUTPUT_FOLDER": PROJECT_ROOT / "datasets" / "vitonHDDataset" / "test" / "output" / "unpaired" / "upper_body"
        },
        ("female", "lower_body"): {
            "UPLOAD_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "lower_body" / "images",
            "CLOTHES_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "lower_body" / "images",
            "OUTPUT_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "lower_body" / "output" / "unpaired" / "lower_body"
        },
        ("female", "dresses"): {
            "UPLOAD_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "dresses" / "images" ,
            "CLOTHES_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "dresses" / "cloth" / "images",
            "OUTPUT_FOLDER": PROJECT_ROOT / "datasets" / "dresscodeDataset" / "dresses" / "output" / "unpaired" / "dresses"
        }
    }

    try:
        paths = paths_dict[(gender, category)]
        UPLOAD_FOLDER = paths["UPLOAD_FOLDER"]
        CLOTHES_FOLDER = paths["CLOTHES_FOLDER"]
        OUTPUT_FOLDER = paths["OUTPUT_FOLDER"]
    except KeyError:
        raise ValueError("Invalid gender or category")

# Function to check if file is an image
def is_image(file):
    try:
        Image.open(file)
        return True
    except:
        return False

def clear_folder_contents(folder_path):
    folder_path = Path(folder_path)
    
    # Iterate over the files and subdirectories in the folder
    for filename in os.listdir(folder_path):
        file_path = folder_path / filename
        # Check if the current item is a file
        if file_path.is_file():
            # Delete the file
            file_path.unlink()
        # If it's a directory, recursively delete its contents and the directory itself
        elif file_path.is_dir():
            # Use shutil.rmtree() to remove the directory and all its contents
            shutil.rmtree(file_path)



# Function to save image to directory
def save_image(file, folder_path, cloth=None):
    folder_path = Path(folder_path)
    file_path = folder_path / file.filename

    # Check if folder exists
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    # Run the renaming logic only if cloth is passed as a boolean
    if cloth is not None:
        base_name, extension = os.path.splitext(file.filename)
        if cloth:
            new_filename = f"{base_name}_1{extension}"
        else:
            new_filename = f"{base_name}_0{extension}"
        file_path = folder_path / new_filename

    # Open the image
    image = Image.open(file)
    # Save the image
    image.save(file_path)


# Combined API endpoint
@app.route('/api/generate_tryon', methods=['POST'])
def upload_and_generate_tryon():
    input_category = request.form['category']
    gender = request.form['gender']
    update_paths(gender, input_category)
    if 'person_image' not in request.files or 'cloth_image' not in request.files:
        return jsonify({'error': 'Images missing in request'}), 400

    person_image = request.files['person_image']
    cloth_image = request.files['cloth_image']

    if person_image.filename == '' or cloth_image.filename == '':
        return jsonify({'error': 'No selected file(s)'}), 400

    if not is_image(person_image.stream) or not is_image(cloth_image.stream):
        return jsonify({'error': 'One or both files are not images'}), 400
    clear_folder_contents(UPLOAD_FOLDER)
    clear_folder_contents(CLOTHES_FOLDER)
    save_image(person_image, UPLOAD_FOLDER, false)
    save_image(cloth_image, CLOTHES_FOLDER, true)

    # # Generate try-on image (assuming main function generates try-on image)
    from inference import main
    main()

    try:
        photo_files = os.listdir(OUTPUT_FOLDER)

        if not photo_files:
            abort(404, "No photos found in the folder")
        elif len(photo_files) > 1:
            abort(500, "More than one photo found in the folder")

        photo_name = photo_files[0]
        photo_path = os.path.join(OUTPUT_FOLDER, photo_name)

        return send_file(photo_path), 200

    except Exception as e:
        # Handle any exceptions here
        abort(500, f"An error occurred: {str(e)}")



    
@app.route('/get_cloth_rec', methods=['GET'])
def get_cloth_rec():
    input_category = request.args.get('category')
    # input_category = request.form['category']
    update_paths_recom(input_category)
    photo_files = os.listdir(Cloth_Recommendation_FOLDER)

    if not photo_files:
        abort(404, "No photos found in the folder")

    photo_id = request.args.get('id')
    if photo_id is None:
        abort(400, "Photo ID is missing in the request")

    try:
        photo_id = int(photo_id)
    except ValueError:
        abort(400, "Photo ID must be an integer")

    photo_files.sort()  # Ensure consistent ordering of photos
    if photo_id < 1 or photo_id > len(photo_files):
        abort(404, "Invalid photo ID")

    photo_name = photo_files[photo_id - 1]
    photo_path = os.path.join(Cloth_Recommendation_FOLDER, photo_name)
    
    # Open the image using PIL
    image = Image.open(photo_path)

    # Resize the image to a compatible size for Flutter (e.g., 300x300)
    resized_image = image.resize((300, 300))

    # Create an in-memory file-like object to hold the resized image data
    image_io = BytesIO()
    resized_image.save(image_io, format='JPEG')
    image_io.seek(0)

    # Send the resized image data
    return send_file(image_io, mimetype='image/jpeg'), 200

@app.route('/', methods=['GET'])
def health():
   
    return jsonify("hello from model api backend")



@app.route('/get_comp_rec', methods=['GET'])
def get_comp_rec():
    # Get the gender parameter from the request
    input_gender = request.args.get('gender')
    update_paths_recom_comp(input_gender)

    # List all photo files in the folder
    photo_files = os.listdir(Cloth_Recommendation_FOLDER_Comp)

    if not photo_files:
        abort(404, "No photos found in the folder")

    # Get the photo ID from the request arguments
    photo_id = request.args.get('id')
    if photo_id is None:
        abort(400, "Photo ID is missing in the request")

    try:
        photo_id = int(photo_id)
    except ValueError:
        abort(400, "Photo ID must be an integer")

    # Ensure consistent ordering of photos
    photo_files.sort()
    if photo_id < 1 or photo_id > len(photo_files):
        abort(404, "Invalid photo ID")

    # Get the photo file based on the photo ID
    photo_name = photo_files[photo_id - 1]
    photo_path = os.path.join(Cloth_Recommendation_FOLDER_Comp, photo_name)

    # Open the image using PIL
    image = Image.open(photo_path)

    # Resize the image to a compatible size for Flutter (e.g., 300x300)
    resized_image = image.resize((300, 400))

    # Create an in-memory file-like object to hold the resized image data
    image_io = BytesIO()
    resized_image.save(image_io, format='JPEG')
    image_io.seek(0)

    # Send the resized image data
    return send_file(image_io, mimetype='image/jpeg'), 200


if __name__ == '__main__':
    app.run(host='192.168.125.111', port=5000, debug=False, threaded=False)
    
    