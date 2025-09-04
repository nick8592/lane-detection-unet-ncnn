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
apt install build-essential git cmake git libopencv-dev -y
```

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
pnnx export/ncnn_models/unet_jit.pt inputshape=[1,3,256,256]
```
This will create `unet_jit.param` and `unet_jit.bin` in `export/ncnn_models/`.

### 3. Deploy with NCNN
Use the generated `.param` and `.bin` files in your NCNN C++/Android/ARM application.
Refer to [NCNN documentation](https://github.com/Tencent/ncnn/wiki) for integration details.

---