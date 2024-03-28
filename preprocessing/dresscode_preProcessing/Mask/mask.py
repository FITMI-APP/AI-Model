from rembg import remove
from PIL import Image
from pathlib import Path
import os


def remove_background(input_cloth_path, output_cloth_path, mask=False):
    input_path = Path(input_cloth_path)

    if input_path.is_file():  # If a single file is provided
        files = [input_path]
    elif input_path.is_dir():  # If a directory is provided
        patterns = ("*.jpg", "*.png", "*.jpeg", "*.webp")
        files = [f for f in input_path.iterdir() if any(f.match(p) for p in patterns)]
    else:
        raise FileNotFoundError(f"The specified input path '{input_cloth_path}' is not valid.")

    # Delete existing images in the output directory
    existing_images = os.listdir(output_cloth_path)
    for existing_image in existing_images:
        os.remove(os.path.join(output_cloth_path, existing_image))

    # Removing background and replacing it with white
    for file in files:
        image_path = str(file)
        input_image = Image.open(image_path)
        output = remove(input_image, only_mask=mask, bgcolor=(255, 255, 255, 255),
                        post_process_mask=True).convert('RGB')
        output.save(output_cloth_path + '/' + file.stem + '.png')
