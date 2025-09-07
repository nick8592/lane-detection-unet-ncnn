"""
UNet with Depthwise Separable Convolutions for efficient lane segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Block.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UNetDepthwiseSmall(nn.Module):
    """
    UNet architecture using Depthwise Separable Convolutions.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.enc1 = DepthwiseSeparableConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DepthwiseSeparableConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DepthwiseSeparableConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DepthwiseSeparableConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DepthwiseSeparableConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final_conv(d1)
        return torch.sigmoid(out)

# --- Model Summary and FLOPs (Optional) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDepthwiseSmall(in_channels=3, out_channels=1).to(device)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Model summary
    from torchinfo import summary
    print(summary(model, input_size=(1, 3, 256, 256), device=device))

    # FLOPs and parameters
    from thop import profile
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Parameters: {params:,}")