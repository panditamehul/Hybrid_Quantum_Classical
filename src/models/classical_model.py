import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

# Set up logging
logger = logging.getLogger('quantum_mri')

class ClassicalMRINet(nn.Module):
    """Classical U-Net based model for MRI reconstruction."""
    
    def __init__(self, stride: int = 2, device: Optional[torch.device] = None):
        """
        Initialize the classical MRI reconstruction model.

        Args:
            stride: Stride for the initial convolution layer
            device: Torch device to use
        """
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        logger.info(f"[ClassicalMRINet] Using device: {self.device}")

        # Simulates quantum downsampling with classical 2x2 conv
        self.conv0 = nn.Conv2d(1, 4, kernel_size=(2, 2), stride=stride)
        self.relu0 = nn.ReLU(inplace=True)

        # Encoder path (5 blocks as per Zhou et al.)
        self.enc1 = self.conv_block(4, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self.conv_block(64, 128)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = self.conv_block(128, 256)  # No pooling after this (bottleneck)

        # Decoder path (4 upsampling blocks)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(256, 128)  # 128 + 128

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)  # 64 + 64

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(64, 32)  # 32 + 32

        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(32, 16)  # 16 + 16

        # Final layers
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        self.final_up = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

        # Move all parameters to device
        self.to(self.device)

    def conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Create a convolutional block with two conv layers, batch norm, and ReLU.

        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels

        Returns:
            nn.Sequential: Convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Reconstructed image
        """
        # Move input to device and store shape
        x = x.to(self.device)
        input_shape = x.shape[2:]

        # Classical first layer
        x = self.relu0(self.conv0(x))

        # Encoder path (5 blocks)
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        enc4 = self.enc4(x)
        x = self.pool4(enc4)

        x = self.enc5(x)  # Bottleneck

        # Decoder path (4 blocks)
        x = self.up1(x)
        x = self._pad_and_concat(x, enc4)
        x = self.dec1(x)

        x = self.up2(x)
        x = self._pad_and_concat(x, enc3)
        x = self.dec2(x)

        x = self.up3(x)
        x = self._pad_and_concat(x, enc2)
        x = self.dec3(x)

        x = self.up4(x)
        x = self._pad_and_concat(x, enc1)
        x = self.dec4(x)

        # Final processing
        x = self.final(x)
        x = self.final_up(x)

        # Final assertion
        assert x.shape[2:] == input_shape, f"Output shape {x.shape[2:]} does not match input shape {input_shape}"

        return x

    def _pad_and_concat(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Pad and concatenate upsampled feature maps with skip connections.

        Args:
            upsampled: Upsampled feature maps
            skip: Skip connection feature maps

        Returns:
            torch.Tensor: Concatenated feature maps
        """
        diff_y = skip.size()[2] - upsampled.size()[2]
        diff_x = skip.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diff_x // 2, diff_x - diff_x // 2,
                                     diff_y // 2, diff_y - diff_y // 2])
        return torch.cat([upsampled, skip], dim=1)
