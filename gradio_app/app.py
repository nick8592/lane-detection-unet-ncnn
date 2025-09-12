import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as T
from models.unet import UNet
from models.unet_depthwise import UNetDepthwise
from models.unet_depthwise_small import UNetDepthwiseSmall
from models.unet_depthwise_nano import UNetDepthwiseNano
from utils.checkpoint import load_checkpoint

MODEL_PATHS = {
    "unet": "../checkpoints/exp_20250908_172015/weights/unet_best.pt",
    "unet_depthwise": "../checkpoints/exp_20250907_172056/weights/unet_best.pt",
    "unet_depthwise_small": "../checkpoints/exp_20250907_094745/weights/unet_best.pt",
    "unet_depthwise_nano": "../checkpoints/exp_20250906_223222/weights/unet_best.pt"
}

IMG_SIZE = 256

MODEL_CLASSES = {
    "unet": UNet,
    "unet_depthwise": UNetDepthwise,
    "unet_depthwise_small": UNetDepthwiseSmall,
    "unet_depthwise_nano": UNetDepthwiseNano
}

def get_model(model_type):
    model_class = MODEL_CLASSES[model_type]
    model = model_class(in_channels=3, out_channels=1)
    checkpoint_path = MODEL_PATHS[model_type]
    load_checkpoint(checkpoint_path, torch.device("cpu"), model)
    model.eval()
    return model

def infer_gradio(model, image):
    # Preprocessing (same as test.py)
    orig_size = image.size
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        output = output.squeeze().cpu().numpy()
        output = np.uint8(output * 255)
    # Resize mask back to original image size
    mask_img = Image.fromarray(output)
    mask_img = mask_img.resize(orig_size, resample=Image.BILINEAR)
    # Invert mask for overlay
    inverted_mask = ImageOps.invert(mask_img.convert("L"))
    color_mask = Image.new("RGBA", orig_size, color=(0, 255, 0, 0))
    alpha = inverted_mask.point(lambda p: int(p * 0.8))
    color_mask.putalpha(alpha)
    image_rgba = image.convert("RGBA")
    overlay_img = Image.alpha_composite(image_rgba, color_mask).convert("RGB")
    return overlay_img, mask_img

def lane_detection(image, model_type):
    model = get_model(model_type)
    return infer_gradio(model, image)

demo = gr.Interface(
    fn=lane_detection,
    inputs=[gr.Image(type="pil"), gr.Radio(["unet", "unet_depthwise", "unet_depthwise_small", "unet_depthwise_nano"], label="Model Type")],
    outputs=[
        gr.Image(type="pil", label="Lane Detection Result (Overlay)"),
        gr.Image(type="pil", label="Mask Output")
    ],
    title="Lane Detection UNet",
    description="Upload a road image and select a model to see lane detection results."
)

if __name__ == "__main__":
    demo.launch()
