#!/usr/bin/env python3
"""
Evaluation script for Quantum MRI Reconstruction.
This script serves as the entry point for evaluating and comparing quantum and classical models.
"""

import os
import argparse
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

from quantum_mri.src.data import DiagnosticNiftiDataset
from quantum_mri.src.models import QuantumMRINet, ClassicalMRINet
from quantum_mri.src.training import compare_models
from quantum_mri.src.utils import plot_model_summary
from quantum_mri.src.config import ConfigManager, ExperimentConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Quantum MRI Reconstruction models')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--quantum-model', type=str, required=True,
                      help='Path to trained quantum model checkpoint')
    parser.add_argument('--classical-model', type=str, required=True,
                      help='Path to trained classical model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu). If None, will use GPU if available')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save evaluation results. If None, uses config output path')
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
    
    # Create output directory
    output_dir = Path(args.output_dir or config['output']['save_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create test data loader
    test_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Load models
    print("\nLoading models...")
    quantum_model = QuantumMRINet(
        noise_config=config['model']['noise_config'],
        encoding_type=config['model']['encoding_type'],
        use_gpu=(device.type == 'cuda'),
        device=device
    )
    quantum_model.load_state_dict(torch.load(args.quantum_model)['model_state_dict'])
    quantum_model.eval()
    
    classical_model = ClassicalMRINet(device=device)
    classical_model.load_state_dict(torch.load(args.classical_model)['model_state_dict'])
    classical_model.eval()
    
    # Compare models
    print("\nComparing models...")
    comparison_results = compare_models(
        quantum_model=quantum_model,
        classical_model=classical_model,
        test_loader=test_loader,
        device=device
    )
    
    # Save comparison results
    results_file = output_dir / 'model_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nComparison results saved to: {results_file}")
    
    # Plot model summaries
    print("\nGenerating model summaries...")
    plot_model_summary(
        quantum_model=quantum_model,
        classical_model=classical_model,
        save_path=str(output_dir / 'model_summaries.png')
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Quantum Model - MSE: {comparison_results['quantum']['mse']:.6f}, "
          f"PSNR: {comparison_results['quantum']['psnr']:.2f} dB, "
          f"SSIM: {comparison_results['quantum']['ssim']:.4f}")
    print(f"Classical Model - MSE: {comparison_results['classical']['mse']:.6f}, "
          f"PSNR: {comparison_results['classical']['psnr']:.2f} dB, "
          f"SSIM: {comparison_results['classical']['ssim']:.4f}")
    print(f"\nImprovements:")
    print(f"MSE: {comparison_results['improvement']['mse']:.2f}%")
    print(f"PSNR: {comparison_results['improvement']['psnr']:.2f}%")
    print(f"SSIM: {comparison_results['improvement']['ssim']:.2f}%")

if __name__ == '__main__':
    main()
