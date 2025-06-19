import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import logging

# Set up logging
logger = logging.getLogger('quantum_mri')

def calculate_metrics(output: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for a single image pair.

    Args:
        output: Model output image
        target: Ground truth image

    Returns:
        Dict containing MSE, PSNR, and SSIM metrics
    """
    mse = np.mean((output - target) ** 2)
    max_val = 1.0
    psnr_val = psnr(target, output, data_range=max_val)
    ssim_val = ssim(target, output, data_range=max_val)
    
    return {
        'mse': mse,
        'psnr': psnr_val,
        'ssim': ssim_val
    }

def compare_models(
    quantum_model: torch.nn.Module,
    classical_model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Compare quantum and classical models on test data.

    Args:
        quantum_model: Quantum model to evaluate
        classical_model: Classical model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Dict containing comparison metrics and visualizations
    """
    quantum_model.eval()
    classical_model.eval()

    # Metrics
    q_metrics = {
        'mse': [],
        'psnr': [],
        'ssim': []
    }

    c_metrics = {
        'mse': [],
        'psnr': [],
        'ssim': []
    }

    # Get some samples for visualization
    vis_samples = []

    with torch.no_grad():
        for batch_idx, (zero_filled, fully_sampled, mask) in enumerate(test_loader):
            zero_filled = zero_filled.to(device)
            fully_sampled = fully_sampled.to(device)

            # Forward pass for both models
            q_outputs = quantum_model(zero_filled)
            c_outputs = classical_model(zero_filled)

            # Resize target to match output sizes if needed
            if q_outputs.shape != fully_sampled.shape:
                q_target = TF.resize(fully_sampled, q_outputs.shape[2:],
                                   interpolation=TF.InterpolationMode.BILINEAR)
            else:
                q_target = fully_sampled

            if c_outputs.shape != fully_sampled.shape:
                c_target = TF.resize(fully_sampled, c_outputs.shape[2:],
                                   interpolation=TF.InterpolationMode.BILINEAR)
            else:
                c_target = fully_sampled

            # Calculate metrics for each sample
            for i in range(zero_filled.size(0)):
                q_output_np = q_outputs[i, 0].cpu().numpy()
                c_output_np = c_outputs[i, 0].cpu().numpy()
                q_target_np = q_target[i, 0].cpu().numpy()
                c_target_np = c_target[i, 0].cpu().numpy()
                input_np = zero_filled[i, 0].cpu().numpy()

                # Calculate metrics for quantum model
                q_metrics_batch = calculate_metrics(q_output_np, q_target_np)
                for metric, value in q_metrics_batch.items():
                    q_metrics[metric].append(value)

                # Calculate metrics for classical model
                c_metrics_batch = calculate_metrics(c_output_np, c_target_np)
                for metric, value in c_metrics_batch.items():
                    c_metrics[metric].append(value)

                # Store some samples for visualization
                if len(vis_samples) < 3 and batch_idx % 2 == 0:
                    # Make sure all samples are the same size for visualization
                    min_h = min(q_output_np.shape[0], c_output_np.shape[0], input_np.shape[0])
                    min_w = min(q_output_np.shape[1], c_output_np.shape[1], input_np.shape[1])

                    # Resize for visualization if needed
                    if input_np.shape != (min_h, min_w):
                        input_np_tensor = torch.from_numpy(input_np).unsqueeze(0)
                        input_np = TF.resize(input_np_tensor, (min_h, min_w)).squeeze(0).numpy()

                    if q_output_np.shape != (min_h, min_w):
                        q_output_np_tensor = torch.from_numpy(q_output_np).unsqueeze(0)
                        q_output_np = TF.resize(q_output_np_tensor, (min_h, min_w)).squeeze(0).numpy()

                    if c_output_np.shape != (min_h, min_w):
                        c_output_np_tensor = torch.from_numpy(c_output_np).unsqueeze(0)
                        c_output_np = TF.resize(c_output_np_tensor, (min_h, min_w)).squeeze(0).numpy()

                    # Store sample
                    vis_samples.append({
                        'input': input_np,
                        'q_output': q_output_np,
                        'c_output': c_output_np,
                        'target': q_target_np if q_target_np.shape == (min_h, min_w) else
                                TF.resize(torch.from_numpy(q_target_np).unsqueeze(0),
                                        (min_h, min_w)).squeeze(0).numpy(),
                        'q_metrics': q_metrics_batch,
                        'c_metrics': c_metrics_batch
                    })

    # Calculate average metrics
    avg_q_metrics = {metric: np.mean(values) for metric, values in q_metrics.items()}
    avg_c_metrics = {metric: np.mean(values) for metric, values in c_metrics.items()}

    # Calculate improvement percentages
    improvement = {
        'mse': (avg_c_metrics['mse'] - avg_q_metrics['mse']) / avg_c_metrics['mse'] * 100,
        'psnr': (avg_q_metrics['psnr'] - avg_c_metrics['psnr']) / avg_c_metrics['psnr'] * 100,
        'ssim': (avg_q_metrics['ssim'] - avg_c_metrics['ssim']) / avg_c_metrics['ssim'] * 100
    }

    # Print comparison results
    print("\nModel Comparison:")
    print(f"  Quantum MSE: {avg_q_metrics['mse']:.6f}, Classical MSE: {avg_c_metrics['mse']:.6f} ({improvement['mse']:.2f}% improvement)")
    print(f"  Quantum PSNR: {avg_q_metrics['psnr']:.2f} dB, Classical PSNR: {avg_c_metrics['psnr']:.2f} dB ({improvement['psnr']:.2f}% improvement)")
    print(f"  Quantum SSIM: {avg_q_metrics['ssim']:.4f}, Classical SSIM: {avg_c_metrics['ssim']:.4f} ({improvement['ssim']:.2f}% improvement)")

    # Create visualizations
    visualize_comparison(vis_samples, avg_q_metrics, avg_c_metrics)

    return {
        'quantum': avg_q_metrics,
        'classical': avg_c_metrics,
        'improvement': improvement
    }

def visualize_comparison(
    vis_samples: List[Dict[str, Any]],
    avg_q_metrics: Dict[str, float],
    avg_c_metrics: Dict[str, float]
) -> None:
    """
    Create visualizations for model comparison.

    Args:
        vis_samples: List of sample images and their metrics
        avg_q_metrics: Average metrics for quantum model
        avg_c_metrics: Average metrics for classical model
    """
    # Create image comparison plot
    plt.figure(figsize=(15, 5 * len(vis_samples)))

    for i, sample in enumerate(vis_samples):
        # Display
        plt.subplot(len(vis_samples), 4, i*4 + 1)
        plt.imshow(sample['target'], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(len(vis_samples), 4, i*4 + 2)
        plt.imshow(sample['input'], cmap='gray')
        plt.title('Zero-Filled Input')
        plt.axis('off')

        plt.subplot(len(vis_samples), 4, i*4 + 3)
        plt.imshow(sample['q_output'], cmap='gray')
        plt.title(f'Quantum Output\nPSNR: {sample["q_metrics"]["psnr"]:.2f}, SSIM: {sample["q_metrics"]["ssim"]:.4f}')
        plt.axis('off')

        plt.subplot(len(vis_samples), 4, i*4 + 4)
        plt.imshow(sample['c_output'], cmap='gray')
        plt.title(f'Classical Output\nPSNR: {sample["c_metrics"]["psnr"]:.2f}, SSIM: {sample["c_metrics"]["ssim"]:.4f}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/model_comparison.png')
    plt.close()

    # Create metrics comparison bar chart
    plt.figure(figsize=(15, 5))

    # MSE (lower is better)
    plt.subplot(1, 3, 1)
    plt.bar(['Quantum', 'Classical'], [avg_q_metrics['mse'], avg_c_metrics['mse']])
    plt.title('MSE (lower is better)')
    plt.ylabel('Mean Squared Error')

    # PSNR (higher is better)
    plt.subplot(1, 3, 2)
    plt.bar(['Quantum', 'Classical'], [avg_q_metrics['psnr'], avg_c_metrics['psnr']])
    plt.title('PSNR (higher is better)')
    plt.ylabel('Peak Signal-to-Noise Ratio (dB)')

    # SSIM (higher is better)
    plt.subplot(1, 3, 3)
    plt.bar(['Quantum', 'Classical'], [avg_q_metrics['ssim'], avg_c_metrics['ssim']])
    plt.title('SSIM (higher is better)')
    plt.ylabel('Structural Similarity Index')

    plt.tight_layout()
    plt.savefig('output/metrics_comparison.png')
    plt.close()
