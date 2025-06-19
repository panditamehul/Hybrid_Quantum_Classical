import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger('quantum_mri')

class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass

class ConfigVersionError(ConfigError):
    """Raised when configuration version is incompatible."""
    pass

class ConfigManager:
    """Manages configuration loading, validation, and versioning."""

    # Current configuration version
    CURRENT_VERSION = "1.0.0"

    # Required configuration structure
    REQUIRED_STRUCTURE = {
        'version': str,
        'data': {
            'nifti_files': list,
            'slice_range': (list, type(None)),
            'intensity_threshold': (float, type(None)),
            'acceleration_factor': (int, float),
            'center_fraction': float,
            'mask_mode': str,
            'scale': (int, float),
            'target_size': (list, tuple)
        },
        'model': {
            'encoding_type': str,
            'noise_config': dict,
            'use_gpu': bool
        },
        'training': {
            'batch_size': int,
            'num_epochs': int,
            'learning_rate': float,
            'early_stopping': dict,
            'optimizer': dict
        }
    }

    # Default configuration values
    DEFAULT_CONFIG = {
        'version': CURRENT_VERSION,
        'data': {
            'slice_range': None,
            'intensity_threshold': None,
            'acceleration_factor': 4,
            'center_fraction': 0.08,
            'mask_mode': 'gaussian',
            'scale': 8.5,
            'target_size': [64, 64]
        },
        'model': {
            'encoding_type': 'amplitude',
            'noise_config': {
                'gate_errors': {
                    'h': 0.001,
                    'ry': 0.002,
                    'rz': 0.002,
                    'cx': 0.01
                }
            },
            'use_gpu': True
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'early_stopping': {
                'patience': 10,
                'delta': 0.001
            },
            'optimizer': {
                'type': 'adam',
                'weight_decay': 0.0001,
                'gradient_clip_val': 1.0
            }
        }
    }

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Optional directory for storing configurations
        """
        self.config_dir = config_dir or os.path.join(os.getcwd(), 'configs')
        os.makedirs(self.config_dir, exist_ok=True)
        self._config_history: List[Dict[str, Any]] = []

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and validate configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dict containing validated configuration parameters

        Raises:
            ConfigError: If configuration loading or validation fails
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load config file: {str(e)}")

        # Validate version
        if 'version' not in config:
            config['version'] = self.CURRENT_VERSION
        elif config['version'] != self.CURRENT_VERSION:
            raise ConfigVersionError(
                f"Configuration version mismatch. Expected {self.CURRENT_VERSION}, "
                f"got {config['version']}"
            )

        # Merge with default configuration
        config = self._merge_with_defaults(config)

        # Validate configuration structure
        self._validate_config(config)

        # Store in history
        self._config_history.append({
            'timestamp': datetime.now().isoformat(),
            'config': config.copy(),
            'source': config_path
        })

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with default values."""
        merged = self.DEFAULT_CONFIG.copy()
        self._recursive_update(merged, config)
        return merged

    def _recursive_update(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively update dictionary with new values."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._recursive_update(base[key], value)
            else:
                base[key] = value

    def _validate_config(self, config: Dict[str, Any]):
        """
        Validate configuration structure and types.

        Args:
            config: Configuration to validate

        Raises:
            ConfigValidationError: If validation fails
        """
        def validate_section(section: Dict[str, Any], required: Dict[str, Any], path: str = ""):
            for key, expected_type in required.items():
                if key not in section:
                    raise ConfigValidationError(f"Missing required key: {path}{key}")
                
                value = section[key]
                if isinstance(expected_type, dict):
                    if not isinstance(value, dict):
                        raise ConfigValidationError(
                            f"Expected dict for {path}{key}, got {type(value)}"
                        )
                    validate_section(value, expected_type, f"{path}{key}.")
                elif isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        raise ConfigValidationError(
                            f"Expected one of {expected_type} for {path}{key}, got {type(value)}"
                        )
                elif not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Expected {expected_type} for {path}{key}, got {type(value)}"
                    )

        validate_section(config, self.REQUIRED_STRUCTURE)

    def save_config(self, config: Dict[str, Any], output_path: str):
        """
        Save configuration to a YAML file.

        Args:
            config: Configuration to save
            output_path: Path to save the configuration
        """
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {str(e)}")

    def get_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Generate a unique hash for the configuration.

        Args:
            config: Configuration to hash

        Returns:
            str: MD5 hash of the configuration
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_config_history(self) -> List[Dict[str, Any]]:
        """
        Get the configuration history.

        Returns:
            List of configuration history entries
        """
        return self._config_history

    def export_config_history(self, output_path: str):
        """
        Export configuration history to a JSON file.

        Args:
            output_path: Path to save the history
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self._config_history, f, indent=2)
            logger.info(f"Exported configuration history to {output_path}")
        except Exception as e:
            raise ConfigError(f"Failed to export configuration history: {str(e)}")

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration.

        Returns:
            Dict containing default configuration
        """
        return self.DEFAULT_CONFIG.copy() 