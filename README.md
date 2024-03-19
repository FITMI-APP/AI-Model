# FitMi AI-Model

Revolutionize the way users experience virtual try-ons with our cutting-edge AI model. Seamlessly integrate this model into your applications to enable users to visualize clothing and accessories in a virtual environment.
> **Abstract**: <br>
>
> 



## Key Features

- **Realistic Garment Rendering:** Witness high-fidelity rendering of clothing items on user images.
- **Pose Adaptation:** Our model adapts clothing to different body poses for a natural and realistic appearance.
- **Compatibility:** Works well with a variety of clothing styles, including tops, bottoms, and dresses.
- **Customization:** Easily integrate and customize the model to fit the unique requirements of your virtual try-on application.

## Usage

Integrating our Virtual Try-On AI model into your app is straightforward. Follow the following detailed instructions to get started quickly.
<details>
<summary><h2>Getting Started</h2></summary>
1. Clone the repository

```sh
git clone https://github.com/FITMI-APP/AI-Model.git
```
2. Install Python dependencies

```sh
conda env create -n FITMI -f FITMI.yml
conda activate FITMI
```
additionally, you should install the required packages manually:
1-install cuda 11.8 [here](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2-Install Microsoft Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
3-set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"  # Update the path accordingly
4-install torch: https://pytorch.org/get-started/locally/ 
or directly write in your command line: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)
5-install cupy : https://docs.cupy.dev/en/stable/install.html#upgrading-cupy 
or directly write in your command line: pip install cupy-cuda11x
6-install cuDNN : https://developer.nvidia.com/rdp/cudnn-archive
copy (lib, include, bin) cuDNN files to the corresponding files in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 respectively
7-pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
</details>

## Citation

If you make use of our work, please cite it using the following format: [Include citation format].



