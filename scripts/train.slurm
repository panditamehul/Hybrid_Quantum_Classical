#!/bin/bash
#SBATCH --job-name=qmri-train
#SBATCH --account=pawsey0309
#SBATCH --partition=work
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=64G

# Load required modules
module load cray-python

# Activate virtual environment (adjust path as needed)
source ~/venvs/qmri/bin/activate

# Set working directory
cd $MYSCRATCH/Hybrid_Quantum_Classical

# Create logs directory
mkdir -p logs

# Run training with Setonix-optimized parameters
python scripts/train.py \
  --config config/example_experiment.yaml \
  --data-dir $MYSCRATCH/data/adnipreprocessed \
  --parallel \
  --num-workers 32 \
  --model-type both

echo "Training completed successfully" 