# FITMI AI-Model

[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LTPovZF6XdsjVqOa_QzbORBjup7YSrjf?usp=sharing)

Revolutionize the way users experience virtual try-ons with our cutting-edge AI model. Seamlessly integrate this model into your applications to enable users to visualize clothing in a virtual environment.
> **Abstract**: <br>
> The word Artificial Intelligence has played a very prominent role, and of late, this term has been gaining much more popularity due to recent advances in the field. Due to these advancements, virtual
> try-on experiences are now made possible. 
> virtual try-on can be achieved by taking an image of a person and a garment taken from a catalog, to dress the person with the given try-on garment and has emerged as a promising solution for consumers
> to virtually try on various items and explore products in their homes or anywhere they want until they find the right choice for them, which is an effective key to the high product return rates, as it
> significantly reduces the likelihood of dissatisfaction post-purchase and by curbing return rates, it promotes eco-friendly shopping practices.
> The challenges any virtual try-on model faces are the pose and texture-style transfer in human body images and lately there are many approaches that overcome these transfer limitations with many fine
> grained differences but often the outputs are low resolution, not photo-realistic or lack texture artifacts, which limits their effectiveness in consumer decision-making.
>
> To enhance the virtual try-on experience,
> we introduce our project, ```FITMI``` that focuses on outperforming these models by overcoming their obstacles and capturing key features related to human body poses
> and garment textures, and by that it ensures accurate outputs, authentic, and immersive try-on experience, thereby empowering consumers to make informed purchasing decisions.




## Key Features

- **Realistic Garment Rendering:** Witness high-fidelity rendering of clothing items on user images.
- **Pose Adaptation:** Our model adapts clothing to different body poses for a natural and realistic appearance.
- **Compatibility:** Works well with a variety of clothing styles, including tops, bottoms, and dresses.
- **Customization:** Easily integrate and customize the model to fit the unique requirements of your virtual try-on application.

## Usage

Integrating our Virtual Try-On AI model into your app is straightforward. Follow the following detailed instructions to get started quickly.
<details>
<summary><h2>Getting Started</h2></summary>

### Installation

1. Clone the repository

```sh
git clone https://github.com/FITMI-APP/AI-Model.git
```
2. Install Python dependencies

```sh
conda env create -n FITMI -f FITMI.yaml
conda activate FITMI
```

### additionally, you should install the required packages manually:

   1. install **[cuda 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)**

   2. Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

   3. set in your environment variables path "C:\Program Files\Microsoft Visual Studio\20xx\Community\VC\Tools\MSVC\14.x.xxxx\bin\Hostx64\x64" **Update the path accordingly**

   4. set in your environment variables "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"   **Update the path accordingly**

   5. install [torch](https://pytorch.org/get-started/locally/)
    or directly type this in your command line:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

   5. install [cupy](https://docs.cupy.dev/en/stable/install.html#upgrading-cupy) or directly type this in your command line:

```sh
pip install cupy-cuda11x
```

   6. install [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

   7. copy (lib, include, bin) cuDNN files to the corresponding files in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 respectively

   8. install xformers

```sh
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```
**note: you may need to reinstall the torch if it got uninstalled from this step**

  9. install detectron2

```sh
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

 10. install these packages
     
```sh
pip install accelerate==0.18.0 diffusers==0.14.0 clean-fid==0.1.35 transformers==4.27.3 torchmetrics rembg ninja av tensorflow scikit-learn huggingface-hub==0.19.4
```

```sh
conda install -c conda-forge opencv
```

### Data Preparation
 #### checkpoints and dataset:
We provide checkpoints for our preprocessing and recommendation system's embeddings in addition to our customized dataset used in our recommendation system. Please download the checkpoints, *.pkl and dataset from our [FITMI](https://fcihelwanedu-my.sharepoint.com/:f:/g/personal/tasnim_mohsen_1375_fci_helwan_edu_eg/Eha1Y-GS-6dEoWSpjwTOTJkBiFYrG-hxAF_5UpnQFc3UAg?e=7glxeS) drive.
Once they are downloaded, the folder structure should look like this:

```
├── AI-Model
|   ├── recommendationSystem
|   |   ├── upper_body
|   |   |   ├── cloth
|   |   |   |   ├── *add upper_body data here*
|   |   |   ├── embedding.pkl
|   |   |   ├── filenames.pkl
|   |   ├── lower_body
|   |   |   ├── cloth
|   |   |   |   ├── *add lower_body data here*
|   |   |   ├── embedding.pkl
|   |   |   ├── filenames.pkl
|   |   ├── dresses
|   |   |   ├── cloth
|   |   |   |   ├── *add dresses data here*
|   |   |   ├── embedding.pkl
|   |   |   ├── filenames.pkl
|   |   ├── complementary
|   |   |   ├── male
|   |   |   |  ├── cloth
|   |   |   |  |   ├── *add male data here*
|   |   |   |  ├── embedding.pkl
|   |   |   |  ├── filenames.pkl
|   |   |   ├── female
|   |   |   |  ├── cloth
|   |   |   |  |   ├── *add female data here*
|   |   |   |  ├── embedding.pkl
|   |   |   |  ├── filenames.pkl
|   ├── preprocessing
|   |   ├── dresscode_preProcessing
|   |   |   ├── DensePose
|   |   |   |   ├── projects
|   |   |   |   |   ├── DensePose
|   |   |   |   |   |   ├── configs
|   |   |   |   |   |   |  ├── model_final_162be9.pkl
|   |   |   ├── OpenPose
|   |   |   |   ├── model
|   |   |   |   |   ├── body_pose_model.pth
|   |   |   |   |   ├── hand_pose_model.pth
|   |   |   |   |   ├── body_pose.caffemodel
|   |   |   |   |   ├── hand_pose.caffemodel
|   |   |   ├── Parsing
|   |   |   |   ├── checkpoints
|   |   |   |   |   ├── exp-schp-201908301523-atr.pth
|   |   ├── vitonHDpreProcessing
|   |   |   ├── DensePose
|   |   |   |   ├── projects
|   |   |   |   |   ├── DensePose
|   |   |   |   |   |   ├── configs
|   |   |   |   |   |   |   ├── model_final_162be9.pkl
|   |   |   ├── OpenPose
|   |   |   |   ├── model
|   |   |   |   |   ├── body_pose_model.pth
|   |   |   |   |   ├── hand_pose_model.pth
|   |   |   |   |   ├── body_pose.caffemodel
|   |   |   |   |   ├── hand_pose.caffemodel
|   |   |   ├── Parsing
|   |   |   |   ├── checkpoints
|   |   |   |   |   ├── exp-schp-201908261155-lip.pth
```

</details>

## Inference

To test the model, run the following command:


```sh
python src/inference.py
```
## Citation

If you make use of our work, please cite it using the following format: [Include citation format].



