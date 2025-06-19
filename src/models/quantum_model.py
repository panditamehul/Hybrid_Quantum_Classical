import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import logging
from joblib import Parallel, delayed
import numpy as np
from ..quantum import (
    QuantumNoiseConfig,
    AmplitudeQuantumCircuit,
    AngleQuantumCircuit
)

# Set up logging
logger = logging.getLogger('quantum_mri')

def process_patch(
    patch: np.ndarray,
    encoding_type: str,
    noise_config: QuantumNoiseConfig,
    use_gpu: bool,
    device: torch.device
) -> np.ndarray:
    """
    Process a single 2x2 patch using quantum circuits.

    Args:
        patch: Input patch as numpy array
        encoding_type: Type of quantum encoding ('amplitude' or 'angle')
        noise_config: Configuration for quantum noise
        use_gpu: Whether to use GPU
        device: Torch device to use

    Returns:
        np.ndarray: Processed patch output

    Raises:
        RuntimeError: If circuit creation or execution fails
    """
    try:
        patch = patch.reshape(2, 2)
        logger.debug(f"Processing patch with shape {patch.shape}")

        # Ensure fresh circuit instance per patch (no shared state between processes)
        if encoding_type == "amplitude":
            circuit = AmplitudeQuantumCircuit(
                patch=patch,
                noise_config=noise_config,
                use_gpu=use_gpu,
                device=device
            )
        else:
            circuit = AngleQuantumCircuit(
                patch=patch,
                noise_config=noise_config,
                use_gpu=use_gpu,
                device=device
            )

        output = circuit.execute()
        logger.debug(f"Successfully processed patch, output shape: {output.shape}")
        return output

    except Exception as e:
        logger.error(f"Error processing patch: {str(e)}")
        raise RuntimeError(f"Failed to process patch: {str(e)}")

class QuantumConvLayer(nn.Module):
    """Quantum convolution layer that processes 2x2 patches using quantum circuits."""
    
    def __init__(self,
                 stride: int = 2,
                 noise_config: Optional[QuantumNoiseConfig] = None,
                 encoding_type: str = "amplitude",
                 use_gpu: bool = False,
                 device: Optional[torch.device] = None,
                 parallel: bool = False,
                 num_workers: int = 4):
        """
        Initialize the quantum convolution layer.

        Args:
            stride: Stride for patch extraction
            noise_config: Configuration for quantum noise
            encoding_type: Type of quantum encoding ('amplitude' or 'angle')
            use_gpu: Whether to use GPU
            device: Torch device to use
            parallel: Whether to use parallel processing
            num_workers: Number of parallel workers
        """
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        else:
            self.device = device
        logger.info(f"[QuantumConvLayer] Using device: {self.device}")
        
        self.stride = stride
        self.encoding_type = encoding_type
        self.use_gpu = use_gpu
        self.parallel = parallel
        self.num_workers = num_workers

        # Use default noise config if none provided
        if noise_config is None:
            noise_config = QuantumNoiseConfig({
                'h': 0.001,    # Hadamard gates
                'ry': 0.002,   # Rotation-Y gates
                'rz': 0.002,   # Rotation-Z gates
                'cx': 0.01     # CNOT gates
            })
        self.noise_config = noise_config

        if self.parallel:
            logger.info(f"[QuantumConvLayer] Parallel processing enabled with {self.num_workers} workers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum convolution layer.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 4, out_height, out_width)
        """
        x = x.to(self.device)
        batch_size, channels, height, width = x.shape
        out_h = (height - 1) // self.stride + 1
        out_w = (width - 1) // self.stride + 1
        out = torch.zeros((batch_size, 4, out_h, out_w), device=self.device)

        for b in range(batch_size):
            patches = []
            patch_positions = []

            # Extract patches
            for i in range(0, height - 1, self.stride):
                for j in range(0, width - 1, self.stride):
                    if i + 2 <= height and j + 2 <= width:
                        patch = x[b, 0, i:i+2, j:j+2].reshape(-1)
                        if patch.shape[0] == 4:
                            patches.append(patch.detach().cpu().numpy())
                            patch_positions.append((i // self.stride, j // self.stride))

            if patches:
                patches_array = np.array(patches)
                logger.debug(f"Processing batch {b} with {len(patches_array)} patches")

                try:
                    if self.parallel:
                        results = Parallel(n_jobs=self.num_workers)(
                            delayed(process_patch)(
                                patch,
                                self.encoding_type,
                                self.noise_config,
                                self.use_gpu,
                                self.device
                            ) for patch in patches_array
                        )
                    else:
                        results = []
                        for patch in patches_array:
                            results.append(
                                process_patch(
                                    patch,
                                    self.encoding_type,
                                    self.noise_config,
                                    self.use_gpu,
                                    self.device
                                )
                            )

                    # Fill the output tensor
                    for idx, output in enumerate(results):
                        pi, pj = patch_positions[idx]
                        out[b, :, pi, pj] = torch.from_numpy(output).to(self.device)

                except Exception as e:
                    logger.error(f"Error processing batch {b}: {str(e)}")
                    raise RuntimeError(f"Failed to process batch {b}: {str(e)}")

        return out

class QuantumMRINet(nn.Module):
    """Quantum-classical hybrid model for MRI reconstruction."""
    
    def __init__(self,
                 noise_config: Optional[QuantumNoiseConfig] = None,
                 encoding_type: str = "amplitude",
                 use_gpu: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initialize the quantum MRI reconstruction model.

        Args:
            noise_config: Configuration for quantum noise
            encoding_type: Type of quantum encoding ('amplitude' or 'angle')
            use_gpu: Whether to use GPU
            device: Torch device to use
        """
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        else:
            self.device = device
        logger.info(f"[QuantumMRINet] Using device: {self.device}")

        # Use default noise config if none provided
        if noise_config is None:
            noise_config = QuantumNoiseConfig({
                'h': 0.001,    # Hadamard gates
                'ry': 0.002,   # Rotation-Y gates
                'rz': 0.002,   # Rotation-Z gates
                'cx': 0.01     # CNOT gates
            })

        # Quantum convolution layer
        self.quantum_conv = QuantumConvLayer(
            stride=2,
            noise_config=noise_config,
            encoding_type=encoding_type,
            use_gpu=use_gpu,
            device=self.device
        )

        # Encoder path (5 blocks as per Zhou et al.)
        self.enc1 = self.conv_block(4, 16)    # 4 channels from quantum output
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

        # Move model to device
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
        x = x.to(self.device)
        
        # Quantum convolution layer
        x = self.quantum_conv(x)

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
