# Quantum MRI Reconstruction

A quantum-classical hybrid model for MRI reconstruction using quantum circuits and classical neural networks. This project implements a novel approach to MRI reconstruction by combining quantum computing techniques with traditional deep learning methods.

## Overview

The project implements a hybrid quantum-classical architecture for MRI reconstruction, where:
- Quantum circuits process 2x2 image patches using amplitude or angle encoding
- Classical U-Net architecture handles the overall image reconstruction
- Variable-density undersampling is used for k-space sampling
- Both quantum and classical models are implemented for comparison

## Project Structure

```
quantum_mri/
├── config/                 # Configuration files
│   ├── base_config.yaml    # Base configuration
│   └── example_experiment.yaml  # Example experiment config
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── data/              # Dataset handling
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   └── utils/             # Helper functions
├── scripts/               # Execution scripts
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantum_mri.git
cd quantum_mri
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses YAML configuration files to manage experiment parameters. Two types of configuration files are provided:

1. `config/base_config.yaml`: Base configuration with default settings
2. `config/example_experiment.yaml`: Example experiment configuration

### Configuration Structure

```yaml
version: "1.0.0"

data:
  nifti_files: []  # List of NIfTI files
  slice_range: null  # Optional: [min, max] slice range
  intensity_threshold: null  # Optional: mean intensity threshold
  acceleration_factor: 4  # K-space undersampling factor
  center_fraction: 0.08  # Center k-space sampling fraction
  mask_mode: "gaussian"  # Sampling pattern
  scale: 8.5  # Scale parameter for sampling
  target_size: [64, 64]  # Target image size

model:
  encoding_type: "amplitude"  # "amplitude" or "angle"
  use_gpu: true
  parallel_processing:
    enabled: false  # Whether to use parallel processing
    num_workers: 4  # Number of parallel workers
  noise_config:
    gate_errors:
      h: 0.001  # Hadamard gates
      ry: 0.002  # Rotation-Y gates
      rz: 0.002  # Rotation-Z gates
      cx: 0.01   # CNOT gates
    qubit_errors: null  # Optional qubit-specific errors

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  early_stopping:
    patience: 10
    delta: 0.001
  optimizer:
    type: "adam"
    weight_decay: 0.0001
    gradient_clip_val: 1.0
  num_workers: 4
  pin_memory: true

output:
  save_path: "output"
  model_name: "quantum_mri"
  tensorboard: true
  save_frequency: 5
```

## Usage

### Training

To train the models:

```bash
python scripts/train.py --config config/example_experiment.yaml --model-type both
```

Options:
- `--config`: Path to configuration file (required)
- `--model-type`: Type of model to train ('quantum', 'classical', or 'both')
- `--device`: Device to use ('cuda' or 'cpu')

### Evaluation

To evaluate trained models:

```bash
python scripts/evaluate.py \
    --config config/example_experiment.yaml \
    --quantum-model output/quantum_mri_quantum_best.pth \
    --classical-model output/quantum_mri_classical_best.pth
```

Options:
- `--config`: Path to configuration file (required)
- `--quantum-model`: Path to trained quantum model checkpoint (required)
- `--classical-model`: Path to trained classical model checkpoint (required)
- `--device`: Device to use ('cuda' or 'cpu')
- `--output-dir`: Directory to save evaluation results

## Features

### Quantum Circuit Features
- Amplitude and angle encoding for image patches
- Configurable quantum noise models
- Support for different quantum gates
- Circuit visualization and debugging tools
- Parallel processing of quantum circuits for improved performance
  - Configurable number of workers
  - Process-level isolation for circuit states
  - Automatic fallback to sequential processing
  - Memory-efficient patch processing

### Classical Features
- U-Net based architecture
- Skip connections for better feature preservation
- Batch normalization and ReLU activation
- Configurable training parameters

### Training Features
- Early stopping
- Learning rate scheduling
- Gradient clipping
- TensorBoard integration
- Model checkpointing

### Evaluation Features
- Comprehensive metrics (MSE, PSNR, SSIM)
- Model comparison visualization
- Detailed performance analysis
- Results export to JSON

## Output

The training and evaluation scripts generate:
1. Model checkpoints
2. Training history plots
3. TensorBoard logs
4. Evaluation metrics
5. Comparison visualizations
6. Model summaries

## Logging

The project uses a comprehensive logging system:
- Console: Brief, user-friendly messages
- File: Detailed logs with timestamps
- TensorBoard: Training metrics and visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Performance Optimization

### Parallel Processing

The quantum circuit processing can be parallelized to improve performance:

1. **Configuration**:
   ```yaml
   model:
     parallel_processing:
       enabled: true  # Enable parallel processing
       num_workers: 8  # Number of parallel workers
   ```

2. **Usage Guidelines**:
   - Enable parallel processing for large datasets or complex circuits
   - Set `num_workers` based on available CPU cores
   - Monitor memory usage when using parallel processing
   - Consider batch size when configuring workers

3. **Best Practices**:
   - Start with `num_workers = CPU cores - 1`
   - Adjust based on memory availability
   - Monitor performance metrics
   - Use TensorBoard to track processing times

4. **Limitations**:
   - GPU usage is not supported with parallel processing
   - Memory usage increases with number of workers
   - Circuit state isolation requires fresh instances
