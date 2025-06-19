#!/bin/bash
#SBATCH --job-name=qmri-eval
#SBATCH --account=pawsey0309
#SBATCH --partition=work
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=32G

# Load required modules
module load cray-python

# Activate virtual environment (adjust path as needed)
source ~/venvs/qmri/bin/activate

# Set working directory
cd $MYSCRATCH/Hybrid_Quantum_Classical

# Create logs directory
mkdir -p logs

# Run evaluation with Setonix-optimized parameters
python scripts/evaluate.py \
  --config config/example_experiment.yaml \
  --quantum-model output/quantum_mri_quantum_best.pth \
  --classical-model output/quantum_mri_classical_best.pth \
  --data-dir $MYSCRATCH/data/adnipreprocessed \
  --parallel \
  --num-workers 32 \
  --output-dir output/evaluation_results

echo "Evaluation completed successfully" 