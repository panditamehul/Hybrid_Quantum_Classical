#!/usr/bin/env python3
"""
Training script for Quantum MRI Reconstruction.
This script serves as the entry point for training both quantum and classical models.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from quantum_mri.src.data import DiagnosticNiftiDataset
from quantum_mri.src.models import QuantumMRINet, ClassicalMRINet
from quantum_mri.src.training import train_model, save_quantum_circuit_info
from quantum_mri.src.utils import plot_training_history
from quantum_mri.src.config import ConfigManager, ExperimentConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train Quantum MRI Reconstruction models')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--model-type', type=str, choices=['quantum', 'classical', 'both'],
                      default='both', help='Type of model to train')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu). If None, will use GPU if available')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    experiment = ExperimentConfig(config)
    
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
            device=device
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
