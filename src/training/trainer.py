import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from ..utils.visualization import plot_training_history

# Set up logging
logger = logging.getLogger('quantum_mri')

def save_quantum_circuit_info(circuit, output_dir: str, prefix: str = 'circuit',
                            writer: Optional[SummaryWriter] = None, step: int = 0) -> None:
    """
    Save quantum circuit diagram and measurement histogram for debugging.

    Args:
        circuit: Quantum circuit to analyze
        output_dir: Directory to save outputs
        prefix: Prefix for output files
        writer: TensorBoard writer for logging
        step: Current training step
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save circuit diagram
    circuit_diagram = circuit_drawer(circuit)
    with open(os.path.join(output_dir, f"{prefix}_diagram.txt"), 'w') as f:
        f.write(str(circuit_diagram))

    # Simulate circuit and get counts
    backend = AerSimulator()
    job = backend.run(circuit, shots=1000)
    result = job.result()
    counts = result.get_counts()

    # Create histogram
    fig = plot_histogram(counts)

    # If writer is provided, log to TensorBoard
    if writer is not None:
        # Convert matplotlib figure to tensor
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = torch.from_numpy(img).permute(2, 0, 1)  # Convert to CxHxW format

        # Log to TensorBoard
        writer.add_image(f'Quantum/{prefix}_histogram', img, step)

    # Save to file
    fig.savefig(os.path.join(output_dir, f"{prefix}_histogram.png"))
    plt.close(fig)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 2,
    lr: float = 0.001,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    model_name: str = "quantum_mri",
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 0.001,
    weight_decay: float = 0.0001,
    gradient_clip_val: float = 1.0
) -> Dict[str, Any]:
    """
    Train a model with early stopping and TensorBoard logging.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on
        save_path: Path to save model checkpoints
        model_name: Name of the model for saving
        early_stopping_patience: Number of epochs to wait for improvement
        early_stopping_delta: Minimum change in validation loss to be considered improvement
        weight_decay: L2 regularization factor
        gradient_clip_val: Maximum gradient norm

    Returns:
        Dict containing training history
    """
    # Create save directory if needed
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard_logs'))

    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare for training
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    best_val_loss = float('inf')
    patience_counter = 0

    # Set up activation hooks for key layers
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks for key layers
    if hasattr(model, 'quantum_conv'):
        model.quantum_conv.register_forward_hook(get_activation('quantum_conv'))
    model.enc1.register_forward_hook(get_activation('enc1'))
    model.enc5.register_forward_hook(get_activation('enc5'))

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, (zero_filled, fully_sampled, _) in enumerate(train_loader):
                zero_filled = zero_filled.to(device)
                fully_sampled = fully_sampled.to(device)

                # Forward pass
                outputs = model(zero_filled)

                # Resize target to match output size if needed
                if outputs.shape != fully_sampled.shape:
                    fully_sampled = TF.resize(fully_sampled, outputs.shape[2:],
                                           interpolation=TF.InterpolationMode.BILINEAR)

                # Calculate loss
                loss = criterion(outputs, fully_sampled)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

                optimizer.step()

                # Update statistics
                train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

                # Log batch-level metrics
                if batch_idx % 10 == 0:  # Log every 10 batches
                    writer.add_scalar('Loss/train_batch', loss.item(),
                                    epoch * len(train_loader) + batch_idx)

                    # Log sample images from the batch
                    if batch_idx == 0:  # Log only from first batch of each epoch
                        for i in range(min(3, zero_filled.size(0))):
                            grid = torch.cat([
                                zero_filled[i],  # Input
                                fully_sampled[i],  # Target
                                outputs[i]  # Prediction
                            ], dim=2)  # Concatenate horizontally
                            writer.add_image(f'Training/Image_{i}', grid, epoch)

        # Calculate average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Log learning rate
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log model parameters and gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'Weights/{name}', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                writer.add_scalar(f'Parameters/{name}_mean', param.data.mean(), epoch)
                writer.add_scalar(f'Parameters/{name}_std', param.data.std(), epoch)

        # Log activations
        for name, activation in activations.items():
            writer.add_histogram(f'Activations/{name}', activation, epoch)

        # Log GPU memory usage if on GPU
        if device.type == 'cuda':
            writer.add_scalar('GPU/Memory_Allocated', torch.cuda.memory_allocated(), epoch)
            writer.add_scalar('GPU/Memory_Cached', torch.cuda.memory_reserved(), epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'mse': [], 'psnr': [], 'ssim': []}
        with torch.no_grad():
            for zero_filled, fully_sampled, _ in val_loader:
                zero_filled = zero_filled.to(device)
                fully_sampled = fully_sampled.to(device)

                # Forward pass
                outputs = model(zero_filled)

                # Resize target to match output size if needed
                if outputs.shape != fully_sampled.shape:
                    fully_sampled = TF.resize(fully_sampled, outputs.shape[2:],
                                           interpolation=TF.InterpolationMode.BILINEAR)

                loss = criterion(outputs, fully_sampled)
                val_loss += loss.item()

                # Calculate metrics for each sample
                for i in range(outputs.size(0)):
                    output_np = outputs[i, 0].cpu().numpy()
                    target_np = fully_sampled[i, 0].cpu().numpy()

                    # Calculate metrics
                    mse = np.mean((output_np - target_np) ** 2)
                    val_metrics['mse'].append(mse)
                    max_val = 1.0
                    val_metrics['psnr'].append(
                        psnr(target_np, output_np, data_range=max_val)
                    )
                    val_metrics['ssim'].append(
                        ssim(target_np, output_np, data_range=max_val)
                    )

        # Calculate average validation loss and metrics
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        avg_metrics = {
            'mse': np.mean(val_metrics['mse']),
            'psnr': np.mean(val_metrics['psnr']),
            'ssim': np.mean(val_metrics['ssim'])
        }
        history['val_metrics'].append(avg_metrics)

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/MSE', avg_metrics['mse'], epoch)
        writer.add_scalar('Metrics/PSNR', avg_metrics['psnr'], epoch)
        writer.add_scalar('Metrics/SSIM', avg_metrics['ssim'], epoch)

        # Log validation images
        if epoch % 5 == 0:  # Log every 5 epochs
            for i in range(min(3, zero_filled.size(0))):
                grid = torch.cat([
                    zero_filled[i],  # Input
                    fully_sampled[i],  # Target
                    outputs[i]  # Prediction
                ], dim=2)  # Concatenate horizontally
                writer.add_image(f'Validation/Image_{i}', grid, epoch)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  MSE: {avg_metrics['mse']:.6f}")
        print(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {avg_metrics['ssim']:.4f}")

        # Early stopping check
        if val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if save_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'metrics': avg_metrics
                }, os.path.join(save_path, f"{model_name}_best.pth"))
                print(f"  Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping triggered after {epoch + 1} epochs")
                break

        # Save checkpoint every epoch
        if save_path is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': avg_metrics
            }, os.path.join(save_path, f"{model_name}_epoch{epoch+1}.pth"))

    # Close TensorBoard writer
    writer.close()

    # Plot training history
    if save_path is not None:
        plot_training_history(history, save_path=os.path.join(save_path, f'{model_name}_training_history.png'))

    return history
