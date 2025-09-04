# UNet
The goal of the UNet is to use BDD100K dataset lane split for lane segmentation. Then use NCNN framework to deploy on the ARM platform.

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
apt install build-essential git cmake wget libprotobuf-dev protobuf-compiler libomp-dev libopencv-dev -y```

### 3. PyTorch Installation
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## BDD100K Dataset

### Directory Tree
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