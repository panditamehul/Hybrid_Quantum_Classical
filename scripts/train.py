#!/usr/bin/env python3
"""
Training script for Quantum MRI Reconstruction.
This script serves as the entry point for training both quantum and classical models.
"""

import os
import argparse
import torch
import logging
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from quantum_mri.src.data import DiagnosticNiftiDataset
from quantum_mri.src.models import QuantumMRINet, ClassicalMRINet
from quantum_mri.src.training import train_model, save_quantum_circuit_info
from quantum_mri.src.utils import plot_training_history
from quantum_mri.src.config import ConfigManager, ExperimentConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Quantum MRI Reconstruction models')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--model-type', type=str, choices=['quantum', 'classical', 'both'],
                      default='both', help='Type of model to train')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu). If None, will use GPU if available')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='Base directory for all NIfTI/mgz files')
    parser.add_argument('--parallel', action='store_true',
                      help='Enable parallel quantum patch processing')
    parser.add_argument('--num-workers', type=int, default=None,
                      help='Override number of parallel workers')
    return parser.parse_args()

def validate_config(config):
    """Validate configuration parameters."""
    if config['model'].get('parallel') and config['model'].get('use_gpu'):
        logger.warning("Parallel quantum patching is CPU-bound. GPU usage may be underutilized.")
    
    num_workers = config['training'].get('num_workers', 4)
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    
    if num_workers > 64:
        logger.warning(f"High worker count ({num_workers}) may cause memory issues")

def log_configuration_summary(config):
    """Log a summary of the current configuration."""
    logger.info("=== Configuration Summary ===")
    logger.info(f"Data directory: {config['data'].get('data_dir', 'Not set')}")
    logger.info(f"Number of files: {len(config['data'].get('nifti_files', []))}")
    logger.info(f"Parallel processing: {config['model'].get('parallel', False)}")
    logger.info(f"Number of workers: {config['training'].get('num_workers', 4)}")
    logger.info(f"GPU usage: {config['model'].get('use_gpu', False)}")
    logger.info("=" * 30)

def merge_cli_config(config, args):
    """Merge CLI arguments with configuration."""
    # Handle data directory
    if args.data_dir:
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
        config['data']['data_dir'] = args.data_dir
        nifti_files = config['data'].get('nifti_files', [])
        
        # Validate file existence after path joining
        valid_files = []
        missing_files = []
        
        for file_path in nifti_files:
            full_path = os.path.join(args.data_dir, file_path)
            if os.path.exists(full_path):
                valid_files.append(full_path)
            else:
                missing_files.append(full_path)
                logger.warning(f"Missing file: {full_path}")
        
        if len(missing_files) > len(valid_files) * 0.5:  # More than 50% missing
            raise FileNotFoundError(f"Too many missing files: {len(missing_files)} out of {len(nifti_files)}")
        
        config['data']['nifti_files'] = valid_files
        logger.info(f"Found {len(valid_files)} valid files out of {len(nifti_files)}")
    
    # Handle parallel processing
    if args.parallel:
        config['model']['parallel'] = True
        logger.info("Parallel quantum patch processing enabled")
    
    # Handle number of workers
    if args.num_workers is not None:
        if args.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {args.num_workers}")
        
        # Setonix-aware CPU detection
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            max_cpus = int(slurm_cpus)
            logger.info(f"Detected SLURM environment: {max_cpus} CPUs allocated")
        else:
            max_cpus = os.cpu_count()
            logger.info(f"Local environment: {max_cpus} CPUs available")
        
        if args.num_workers > max_cpus:
            logger.warning(f"num_workers ({args.num_workers}) > available CPUs ({max_cpus})")
        
        if args.num_workers > 64:
            logger.warning("High number of workers may lead to memory contention. Ensure this matches SLURM settings.")
        
        config['training']['num_workers'] = args.num_workers
        logger.info(f"Using {args.num_workers} parallel workers")

def main():
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    experiment = ExperimentConfig(config)
    
    # Merge CLI arguments with configuration
    merge_cli_config(config, args)
    
    # Validate configuration
    validate_config(config)
    
    # Log configuration summary
    log_configuration_summary(config)
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = DiagnosticNiftiDataset(
        nifti_files=config['data']['nifti_files'],
        slice_range=config['data'].get('slice_range'),
        intensity_threshold=config['data'].get('intensity_threshold'),
        acceleration_factor=config['data']['acceleration_factor'],
        center_fraction=config['data']['center_fraction'],
        mask_mode=config['data']['mask_mode'],
        scale=config['data']['scale'],
        target_size=config['data']['target_size']
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Create output directory
    output_dir = Path(config['output']['save_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    if args.model_type in ['quantum', 'both']:
        print("\nTraining Quantum Model...")
        quantum_model = QuantumMRINet(
            noise_config=config['model']['noise_config'],
            encoding_type=config['model']['encoding_type'],
            use_gpu=(device.type == 'cuda'),
            device=device,
            parallel=config['model'].get('parallel', False),
            num_workers=config['training'].get('num_workers', 4)
        )
        
        q_history = train_model(
            model=quantum_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            lr=config['training']['learning_rate'],
            device=device,
            save_path=str(output_dir),
            model_name=f"{config['output']['model_name']}_quantum",
            early_stopping_patience=config['training']['early_stopping']['patience'],
            early_stopping_delta=config['training']['early_stopping']['delta'],
            weight_decay=config['training']['optimizer']['weight_decay'],
            gradient_clip_val=config['training']['optimizer']['gradient_clip_val']
        )
        
        # Plot training history
        plot_training_history(
            q_history,
            save_path=str(output_dir / f'quantum_training_history_{config["model"]["encoding_type"]}.png')
        )
    
    if args.model_type in ['classical', 'both']:
        print("\nTraining Classical Model...")
        classical_model = ClassicalMRINet(device=device)
        
        c_history = train_model(
            model=classical_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            lr=config['training']['learning_rate'],
            device=device,
            save_path=str(output_dir),
            model_name=f"{config['output']['model_name']}_classical",
            early_stopping_patience=config['training']['early_stopping']['patience'],
            early_stopping_delta=config['training']['early_stopping']['delta'],
            weight_decay=config['training']['optimizer']['weight_decay'],
            gradient_clip_val=config['training']['optimizer']['gradient_clip_val']
        )
        
        # Plot training history
        plot_training_history(
            c_history,
            save_path=str(output_dir / 'classical_training_history.png')
        )

if __name__ == '__main__':
    main()
