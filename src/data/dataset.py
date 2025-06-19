import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from typing import List, Tuple, Optional, Dict, Any
import logging

# Set up logging
logger = logging.getLogger('quantum_mri')

class DataProcessingError(Exception):
    """Raised when there's an error in data processing or loading."""
    pass

def get_sampling_prob(ny: int, mode: str = 'gaussian', scale: Optional[float] = None) -> np.ndarray:
    """
    Get sampling probability distribution along phase-encode dimension.

    Args:
        ny: Number of phase-encode lines
        mode: Sampling mode ('gaussian', 'linear', or 'uniform')
        scale: Scale parameter for gaussian mode

    Returns:
        np.ndarray: Probability distribution for sampling
    """
    y_coords = np.arange(ny)
    center = ny / 2
    if mode == 'gaussian':
        std = ny / 6 if scale is None else ny / scale
        prob = np.exp(-((y_coords - center) ** 2) / (2 * std ** 2))
    elif mode == 'linear':
        prob = 1 - np.abs(y_coords - center) / center
    elif mode == 'uniform':
        prob = np.ones(ny)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    prob /= prob.sum()
    return prob

def generate_variable_density_mask(
    ny: int,
    acceleration: int,
    center_fraction: float = 0.08,
    mode: str = 'gaussian',
    scale: Optional[float] = None,
    seed: Optional[int] = None,
    visualize: bool = False
) -> np.ndarray:
    """
    Generate a variable-density undersampling mask for k-space.

    Args:
        ny: Number of phase-encode lines
        acceleration: Undersampling factor
        center_fraction: Fraction of center k-space to fully sample
        mode: Sampling mode ('gaussian', 'linear', or 'uniform')
        scale: Scale parameter for gaussian mode
        seed: Random seed for reproducibility
        visualize: Whether to visualize the mask

    Returns:
        np.ndarray: 2D undersampling mask
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.zeros(ny, dtype=bool)

    # Always sample center of k-space
    center_lines = int(ny * center_fraction)
    total_lines = ny // acceleration

    if total_lines <= center_lines:
        total_lines = center_lines

    center_start = (ny - center_lines) // 2
    mask[center_start:center_start + center_lines] = True

    # Sample remaining lines based on probability distribution
    remaining = total_lines - center_lines
    prob = get_sampling_prob(ny, mode=mode, scale=scale)

    if remaining > 0:
        y_coords = np.arange(ny)
        outside_center = np.setdiff1d(y_coords, np.arange(center_start, center_start + center_lines))
        prob_outside = prob[outside_center]
        prob_outside /= prob_outside.sum()
        chosen = np.random.choice(outside_center, size=remaining, replace=False, p=prob_outside)
        mask[chosen] = True

    final_sampled = np.sum(mask)
    if visualize:
        print(f"[INFO] Accel={acceleration} | Center={center_lines} | Random={remaining} | Total={final_sampled} / {ny}")
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        plt.plot(prob)
        plt.title(f"Sampling Probability ({mode})")
        plt.xlabel("Phase-Encode Line")
        plt.ylabel("Probability")

        plt.subplot(1, 2, 2)
        plt.imshow(mask.reshape(ny, 1).T, cmap='gray', aspect='auto')
        plt.title("Mask (1D)")
        plt.yticks([])
        plt.xlabel("Phase-Encode Line")
        plt.tight_layout()
        plt.show()

    return mask.reshape(ny, 1)

class DiagnosticNiftiDataset(Dataset):
    """
    A dataset class for NIfTI files with smart slice filtering and downsampling for faster processing.
    """
    def __init__(self,
                 nifti_files: List[str],
                 slice_range: Optional[Tuple[int, int]] = None,
                 acceleration_factor: int = 4,
                 center_fraction: float = 0.08,
                 mask_mode: str = 'gaussian',
                 scale: float = 8.5,
                 target_size: Tuple[int, int] = (64, 64),
                 intensity_threshold: Optional[float] = None,
                 plot_slices: bool = False):
        """
        Initialize the dataset.

        Args:
            nifti_files: List of paths to NIfTI files
            slice_range: Tuple (min_slice, max_slice) or None to use smart filtering
            acceleration_factor: Undersampling factor (2x or 4x)
            center_fraction: Fraction of center k-space to fully sample
            mask_mode: Type of sampling pattern ('gaussian', 'linear', 'uniform')
            scale: Scale parameter for variable density sampling
            target_size: Target size to resize images to (e.g., (64, 64))
            intensity_threshold: If set, uses mean slice intensity > threshold for selection
            plot_slices: If True, plot sample slices from the selected ones
        """
        self.nifti_files = nifti_files
        self.slice_range = slice_range
        self.acceleration_factor = acceleration_factor
        self.center_fraction = center_fraction
        self.mask_mode = mask_mode
        self.scale = scale
        self.target_size = target_size
        self.intensity_threshold = intensity_threshold
        self.plot_slices = plot_slices

        # Load all NIfTI files once and keep them in memory
        self.nifti_data = {}
        self.slices = []
        self._slice_cache = {}  # Cache for filtered slice indices

        print(f"Loading {len(nifti_files)} NIfTI files...")
        for file_idx, file_path in enumerate(nifti_files):
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
                self.nifti_data[file_path] = data
                n_slices = data.shape[-1]

                if intensity_threshold is not None:
                    # Check cache first
                    cache_key = f"{file_path}_{intensity_threshold}"
                    if cache_key in self._slice_cache:
                        selected = self._slice_cache[cache_key]
                    else:
                        # Normalize entire volume
                        norm_data = (data - data.min()) / (data.max() - data.min() + 1e-5)
                        selected = [i for i in range(n_slices)
                                  if np.mean(norm_data[:, :, i]) > intensity_threshold]
                        self._slice_cache[cache_key] = selected

                    print(f"  {file_path} — Selected {len(selected)} slices with threshold {intensity_threshold}")

                    if plot_slices and len(selected) > 0:
                        n_plot = min(5, len(selected))
                        fig, axs = plt.subplots(1, n_plot, figsize=(15, 3))
                        for j in range(n_plot):
                            axs[j].imshow(norm_data[:, :, selected[j]], cmap='gray')
                            axs[j].set_title(f"Slice {selected[j]}")
                            axs[j].axis('off')
                        plt.tight_layout()
                        plt.savefig(f'output/selected_slices_{file_idx}.png')
                        plt.close()

                    for slice_idx in selected:
                        self.slices.append((file_idx, slice_idx))

                else:
                    # Use original slice range logic if no threshold specified
                    min_slice = 0 if slice_range is None else max(0, slice_range[0])
                    max_slice = n_slices if slice_range is None else min(n_slices, slice_range[1])
                    for slice_idx in range(min_slice, max_slice):
                        self.slices.append((file_idx, slice_idx))

            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                raise DataProcessingError(f"Failed to load NIfTI file {file_path}: {str(e)}")

        print(f"Total slices: {len(self.slices)}")

    def _undersample_kspace(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an undersampled version of the image by applying
        a variable density mask in k-space.

        Args:
            image: Input image to undersample

        Returns:
            Tuple[np.ndarray, np.ndarray]: Undersampled image and sampling mask
        """
        ny, nx = image.shape

        # Generate the mask using the improved method
        mask_1d = generate_variable_density_mask(
            ny,
            self.acceleration_factor,
            center_fraction=self.center_fraction,
            mode=self.mask_mode,
            scale=self.scale,
            seed=None  # Different mask per slice for diversity
        )

        # Expand to 2D
        mask_2d = np.repeat(mask_1d, nx, axis=1)

        # Convert to k-space using FFT
        kspace = np.fft.fftshift(np.fft.fft2(image))

        # Apply mask
        masked_kspace = kspace * mask_2d

        # Convert back to image domain using inverse FFT
        zero_filled = np.abs(np.fft.ifft2(np.fft.ifftshift(masked_kspace)))

        return zero_filled, mask_2d

    def __len__(self) -> int:
        """Return the total number of slices."""
        return len(self.slices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset with resizing.

        Args:
            idx: Index of the sample to get

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Zero-filled image, fully sampled image, and mask
        """
        file_idx, slice_idx = self.slices[idx]
        file_path = self.nifti_files[file_idx]
        # Use the in-memory data instead of loading the file again
        data = self.nifti_data[file_path]
        slice_data = data[:, :, slice_idx]
        # Vectorized normalization: normalize the entire slice at once
        slice_min = slice_data.min()
        slice_max = slice_data.max()
        if slice_max > slice_min:  # Avoid division by zero
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        # Create undersampled version before resizing
        zero_filled, mask = self._undersample_kspace(slice_data)
        # Vectorized resizing: convert to tensors and resize in one go
        if self.target_size is not None:
            # Convert to tensors for resizing
            slice_tensor = torch.from_numpy(slice_data).float().unsqueeze(0)
            zero_filled_tensor = torch.from_numpy(zero_filled).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            # Resize all tensors at once
            slice_tensor = TF.resize(slice_tensor, self.target_size,
                                   interpolation=TF.InterpolationMode.BILINEAR)
            zero_filled_tensor = TF.resize(zero_filled_tensor, self.target_size,
                                         interpolation=TF.InterpolationMode.BILINEAR)
            mask_tensor = TF.resize(mask_tensor, self.target_size,
                                  interpolation=TF.InterpolationMode.NEAREST)
            # Convert back to numpy arrays
            slice_data = slice_tensor.squeeze(0).numpy()
            zero_filled = zero_filled_tensor.squeeze(0).numpy()
            mask = mask_tensor.squeeze(0).numpy()
        # Convert to tensors
        fully_sampled = torch.from_numpy(slice_data).float().unsqueeze(0)
        zero_filled = torch.from_numpy(zero_filled).float().unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return zero_filled, fully_sampled, mask
