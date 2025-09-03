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