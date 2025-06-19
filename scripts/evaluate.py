#!/usr/bin/env python3
"""
Evaluation script for Quantum MRI Reconstruction.
This script serves as the entry point for evaluating and comparing quantum and classical models.
"""

import os
import argparse
import torch
import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from quantum_mri.src.data import DiagnosticNiftiDataset
from quantum_mri.src.models import QuantumMRINet, ClassicalMRINet
from quantum_mri.src.training import compare_models
from quantum_mri.src.utils import plot_model_summary
from quantum_mri.src.config import ConfigManager, ExperimentConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        device=device,
        parallel=config['model'].get('parallel', False),
        num_workers=config['training'].get('num_workers', 4)
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
