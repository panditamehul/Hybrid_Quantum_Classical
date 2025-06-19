import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
import logging

# Set up logging
logger = logging.getLogger('quantum_mri')

def plot_training_history(history: Dict[str, List[Any]], save_path: Optional[str] = None) -> None:
    """
    Plot training history including loss and metrics.

    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MSE
    plt.subplot(2, 2, 2)
    plt.plot([m['mse'] for m in history['val_metrics']])
    plt.title('MSE on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')

    # Plot PSNR
    plt.subplot(2, 2, 3)
    plt.plot([m['psnr'] for m in history['val_metrics']])
    plt.title('PSNR on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')

    # Plot SSIM
    plt.subplot(2, 2, 4)
    plt.plot([m['ssim'] for m in history['val_metrics']])
    plt.title('SSIM on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")

    plt.close()

def visualize_dataset_samples(
    dataset: Any,
    num_samples: int = 3,
    save_path: Optional[str] = None
) -> None:
    """
    Display samples from the dataset to verify preprocessing.

    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to display
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 5*num_samples))

    for i in range(num_samples):
        try:
            zero_filled, fully_sampled, mask = dataset[i]

            # Convert tensors to numpy arrays for visualization
            zero_filled = zero_filled.squeeze().numpy()
            fully_sampled = fully_sampled.squeeze().numpy()
            mask = mask.squeeze().numpy()

            # Calculate error map
            error = np.abs(fully_sampled - zero_filled)

            # Calculate metrics
            from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
            ssim_val = ssim(fully_sampled, zero_filled, data_range=1.0)
            psnr_val = psnr(fully_sampled, zero_filled, data_range=1.0)

            # Display
            plt.subplot(num_samples, 4, i*4 + 1)
            plt.imshow(fully_sampled, cmap='gray')
            plt.title('Fully Sampled')
            plt.axis('off')

            plt.subplot(num_samples, 4, i*4 + 2)
            plt.imshow(zero_filled, cmap='gray')
            plt.title(f'Zero-Filled (PSNR: {psnr_val:.2f})')
            plt.axis('off')

            plt.subplot(num_samples, 4, i*4 + 3)
            plt.imshow(error, cmap='hot')
            plt.title(f'Error (SSIM: {ssim_val:.4f})')
            plt.axis('off')

            plt.subplot(num_samples, 4, i*4 + 4)
            plt.imshow(mask, cmap='gray')
            plt.title('k-space Mask')
            plt.axis('off')

        except IndexError:
            break

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        logger.info(f"Dataset samples plot saved to {save_path}")
    else:
        plt.savefig('output/dataset_samples.png')
        logger.info("Dataset samples plot saved to output/dataset_samples.png")
    
    plt.close()

def plot_model_summary(
    quantum_model: Any,
    classical_model: Any,
    save_path: Optional[str] = None
) -> None:
    """
    Print and visualize model architecture summaries.

    Args:
        quantum_model: Quantum model to summarize
        classical_model: Classical model to summarize
        save_path: Optional path to save the summary
    """
    from torchsummary import summary

    print("\n🔍 MODEL ARCHITECTURE SUMMARY")
    print("="*40)
    print("\n📦 Quantum Model Summary:")
    try:
        summary(quantum_model, (1, 64, 64))
    except:
        print("  (torchsummary not installed or error in quantum layer — skip)")

    print("\n📦 Classical Model Summary:")
    try:
        summary(classical_model, (1, 64, 64))
    except:
        print("  (torchsummary not installed — skip)")

    if save_path is not None:
        # Save model summaries to text file
        with open(save_path, 'w') as f:
            f.write("MODEL ARCHITECTURE SUMMARY\n")
            f.write("="*40 + "\n\n")
            
            f.write("Quantum Model Summary:\n")
            f.write("-"*20 + "\n")
            try:
                f.write(str(summary(quantum_model, (1, 64, 64), verbose=0)))
            except:
                f.write("(torchsummary not installed or error in quantum layer — skip)\n")
            
            f.write("\nClassical Model Summary:\n")
            f.write("-"*20 + "\n")
            try:
                f.write(str(summary(classical_model, (1, 64, 64), verbose=0)))
            except:
                f.write("(torchsummary not installed — skip)\n")
        
        logger.info(f"Model summary saved to {save_path}")
