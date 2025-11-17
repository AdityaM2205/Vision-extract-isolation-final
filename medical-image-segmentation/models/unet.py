import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SimpleUNet(nn.Module):
    """A simpler U-Net implementation for general-purpose segmentation"""
    def __init__(self, in_channels: int = 3, out_channels: int = 1, 
                 features: List[int] = [64, 128, 256, 512], bilinear: bool = True) -> None:
        super(SimpleUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        features = features[::-1]
        
        for i in range(len(features) - 1):
            self.decoder.append(
                Up(features[i], features[i+1] // 2, bilinear)
            )
        
        # Final convolution
        self.final_conv = OutConv(features[-1] // 2, out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for idx, up in enumerate(self.decoder):
            x = up(x, skip_connections[idx+1])
        
        # Final convolution
        x = self.final_conv(x)
        return self.activation(x)

class UNet(nn.Module):
    """U-Net architecture with more capacity"""
    def __init__(self, in_channels: int = 3, out_channels: int = 1, 
                 features: List[int] = [64, 128, 256, 512, 1024], bilinear: bool = False) -> None:
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Initial double conv
        self.initial_conv = DoubleConv(in_channels, features[0])
        in_channels = features[0]
        
        # Downsampling through the model
        for feature in features[1:]:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        features = features[::-1]
        
        for i in range(len(features) - 1):
            self.decoder.append(
                Up(features[i], features[i+1] // 2, bilinear)
            )
        
        # Final convolution
        self.final_conv = OutConv(features[-1] // 2, out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder
        skip_connections = [x]
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        x = skip_connections[0]  # Start with the bottleneck
        
        for idx, up in enumerate(self.decoder):
            x = up(x, skip_connections[idx+1])
        
        # Final convolution
        x = self.final_conv(x)
        return self.activation(x)
