from typing import Dict, Any, Optional, List, Union
import os
import json
from datetime import datetime
import logging
from pathlib import Path
from .config_manager import ConfigManager, ConfigError

logger = logging.getLogger('quantum_mri')

class ExperimentConfig:
    """Manages experiment-specific configuration and metadata."""

    def __init__(self, config: Dict[str, Any], experiment_name: Optional[str] = None):
        """
        Initialize experiment configuration.

        Args:
            config: Base configuration dictionary
            experiment_name: Optional name for the experiment
        """
        self.config = config
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'config_version': config.get('version', 'unknown'),
            'config_hash': ConfigManager().get_config_hash(config)
        }

    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config['data']

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config['model']

    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config['training']

    def update_metadata(self, key: str, value: Any):
        """
        Update experiment metadata.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def add_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """
        Add a metric to the experiment metadata.

        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional step number
        """
        if 'metrics' not in self.metadata:
            self.metadata['metrics'] = {}
        
        if metric_name not in self.metadata['metrics']:
            self.metadata['metrics'][metric_name] = []
        
        self.metadata['metrics'][metric_name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })

    def save_experiment(self, output_dir: str):
        """
        Save experiment configuration and metadata.

        Args:
            output_dir: Directory to save experiment data
        """
        try:
            # Create experiment directory
            experiment_dir = os.path.join(output_dir, self.experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Save configuration
            config_path = os.path.join(experiment_dir, 'config.yaml')
            ConfigManager().save_config(self.config, config_path)

            # Save metadata
            metadata_path = os.path.join(experiment_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"Saved experiment data to {experiment_dir}")
        except Exception as e:
            raise ConfigError(f"Failed to save experiment: {str(e)}")

    @classmethod
    def load_experiment(cls, experiment_dir: str) -> 'ExperimentConfig':
        """
        Load experiment from directory.

        Args:
            experiment_dir: Directory containing experiment data

        Returns:
            ExperimentConfig instance
        """
        try:
            # Load configuration
            config_path = os.path.join(experiment_dir, 'config.yaml')
            config = ConfigManager().load_config(config_path)

            # Load metadata
            metadata_path = os.path.join(experiment_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Create experiment instance
            experiment = cls(config, os.path.basename(experiment_dir))
            experiment.metadata = metadata

            return experiment
        except Exception as e:
            raise ConfigError(f"Failed to load experiment: {str(e)}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the experiment configuration.

        Returns:
            Dict containing experiment summary
        """
        return {
            'experiment_name': self.experiment_name,
            'created_at': self.metadata['created_at'],
            'config_version': self.metadata['config_version'],
            'data': {
                'nifti_files': len(self.data_config['nifti_files']),
                'acceleration_factor': self.data_config['acceleration_factor'],
                'target_size': self.data_config['target_size']
            },
            'model': {
                'encoding_type': self.model_config['encoding_type'],
                'use_gpu': self.model_config['use_gpu']
            },
            'training': {
                'batch_size': self.training_config['batch_size'],
                'num_epochs': self.training_config['num_epochs'],
                'learning_rate': self.training_config['learning_rate']
            }
        }

    def __str__(self) -> str:
        """Get string representation of experiment configuration."""
        summary = self.get_summary()
        return json.dumps(summary, indent=2) 