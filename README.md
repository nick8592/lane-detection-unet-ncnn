# UNet Lane Segmentation & NCNN Deployment

This repository provides a complete pipeline for lane segmentation using a UNet model trained on the BDD100K dataset. It includes modular scripts for training, validation, and inference, with configuration-driven workflows and TensorBoard logging. The project supports exporting PyTorch models for deployment with the NCNN framework, enabling efficient inference on ARM and x86 platforms via C++.

Features:
- UNet architecture for lane segmentation
- BDD100K dataset integration
- Configurable training and inference scripts
- Checkpointing and metrics
- TensorBoard support
- Model export for NCNN (TorchScript → pnnx → NCNN)
- C++ deployment example with NCNN and OpenCV

---

## Setup Instructions

### 1. Quick Start (Docker)
```bash
docker run -it --gpus all --shm-size=16g -v /home:/home <image_id>
```

### 2. Environment Setup
```bash
apt update && apt upgrade -y
apt install python3 python3-pip -y
apt install build-essential git cmake wget libprotobuf-dev protobuf-compiler libomp-dev libopencv-dev -y
```

### 3. PyTorch Installation
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

---

## Dataset

### BDD100K
#### Directory Tree
```
bdd100k
├── images
│   └── 100k
│       ├── test
│       ├── train
│       └── val
└── lane
    ├── colormaps
    ├── masks
    │   ├── traind
    │   └── val
    └── polygons

```

## Training Usage

### 1. Edit Training Config
Edit `config/train_config.yaml` to set dataset paths, batch size, epochs, learning rate, image size, and model parameters.

### 2. Run Training
```bash
python3 scripts/train.py
```
Checkpoints and logs will be saved in the `checkpoints/` and `runs/` folders.

TensorBoard logs are available in `runs/`. To view training progress:
```bash
tensorboard --logdir runs/
```

## Inference Usage

### 1. Edit Inference Config
Edit `config/inference_config.yaml` to set the trained checkpoint, input images directory, output directory, and model parameters.

### 2. Run Inference
```bash
python3 scripts/inference.py
```
Output masks will be saved in the specified output directory. Progress is shown with tqdm.

---

## Exporting Model for NCNN Deployment

### 1. Export PyTorch Model to TorchScript
Run the export script to convert your trained UNet model to TorchScript:
```bash
python export/export_to_pnnx.py
```
This will generate `unet_jit.pt` in `export/ncnn_models/`.

### 2. Convert TorchScript to NCNN Format with pnnx
Run pnnx to generate NCNN model files:
```bash
pnnx ncnn_deploy/ncnn_models/unet_jit.pt inputshape=[1,3,256,256]
```
This will create `unet_jit.param` and `unet_jit.bin` in `export/ncnn_models/`.

### 3. Deploy with NCNN
Use the generated `.param` and `.bin` files in your NCNN C++/Android/ARM application.
Refer to [NCNN documentation](https://github.com/Tencent/ncnn/wiki) for integration details.

## NCNN Library Setup

To build NCNN from source:
```bash
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
```

## NCNN C++ Deployment

### 1. Build NCNN C++ Inference Code
Go to the deployment folder:
```bash
cd ncnn_deploy
mkdir build && cd build
cmake ..
make -j$(nproc)
```
If you see errors about missing NCNN or OpenCV, make sure to set the correct paths in `CMakeLists.txt`:
```cmake
set(ncnn_DIR "/path/to/ncnn/install/lib/cmake/ncnn")
include_directories("/path/to/ncnn/install/include")
```

### 2. Run Inference
```bash
./unet_ncnn <input_image>
```
This will save the output mask as `output_mask.png`.

### 3. Troubleshooting
- If CMake cannot find protobuf, try:
  ```bash
  cmake .. -DProtobuf_INCLUDE_DIR=/usr/include -DProtobuf_LIBRARIES=/usr/lib/x86_64-linux-gnu/libprotobuf.so
  ```
- If CMake cannot find NCNN, set `ncnn_DIR` to the folder containing `ncnnConfig.cmake`.
- For ARM cross-compilation, set toolchain and paths as needed.

---