import logging
import json
from typing import Dict, Optional
from functools import lru_cache
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Set up logging
logger = logging.getLogger('quantum_mri')

class QuantumNoiseConfig:
    """Configuration class for quantum noise parameters."""

    # Supported gate types
    SINGLE_QUBIT_GATES = {'h', 'ry', 'rz'}
    TWO_QUBIT_GATES = {'cx'}
    SUPPORTED_GATES = SINGLE_QUBIT_GATES | TWO_QUBIT_GATES

    def __init__(self,
                 gate_errors: Dict[str, float],
                 qubit_errors: Optional[Dict[int, Dict[str, float]]] = None):
        """
        Initialize noise configuration.

        Args:
            gate_errors: Dictionary mapping gate names to error probabilities
            qubit_errors: Optional dictionary mapping qubit indices to gate-specific errors

        Raises:
            ValueError: If gate names are not supported or error rates are invalid
        """
        # Validate gate names and error rates
        self._validate_gate_errors(gate_errors)
        if qubit_errors:
            self._validate_qubit_errors(qubit_errors)

        self.gate_errors = gate_errors
        self.qubit_errors = qubit_errors or {}

        logger.debug(f"Created noise config with {len(gate_errors)} global gates and "
                    f"{len(qubit_errors) if qubit_errors else 0} qubit-specific errors")

    def _validate_gate_errors(self, gate_errors: Dict[str, float]):
        """Validate gate names and error rates."""
        for gate, error_rate in gate_errors.items():
            if gate not in self.SUPPORTED_GATES:
                raise ValueError(f"Unsupported gate type: {gate}. "
                               f"Supported gates are: {self.SUPPORTED_GATES}")
            if not 0 <= error_rate <= 1:
                raise ValueError(f"Invalid error rate for gate {gate}: {error_rate}. "
                               f"Error rates must be between 0 and 1")
            if error_rate > 0.1:  # Warning for high error rates
                logger.warning(f"High error rate ({error_rate}) specified for gate {gate}")

    def _validate_qubit_errors(self, qubit_errors: Dict[int, Dict[str, float]]):
        """Validate qubit-specific error configurations."""
        for qubit_idx, qubit_gate_errors in qubit_errors.items():
            if not isinstance(qubit_idx, int) or qubit_idx < 0:
                raise ValueError(f"Invalid qubit index: {qubit_idx}. Must be a non-negative integer")
            self._validate_gate_errors(qubit_gate_errors)

    def __hash__(self):
        """Create a hash of the configuration for caching."""
        return hash(json.dumps({
            'gate_errors': self.gate_errors,
            'qubit_errors': self.qubit_errors
        }, sort_keys=True))

    def __eq__(self, other):
        """Enable comparison of noise configurations."""
        if not isinstance(other, QuantumNoiseConfig):
            return False
        return (self.gate_errors == other.gate_errors and
                self.qubit_errors == other.qubit_errors)

    def get_summary(self) -> str:
        """Get a human-readable summary of the noise configuration."""
        summary = ["Noise Configuration Summary:"]
        summary.append("Global Gate Errors:")
        for gate, error in sorted(self.gate_errors.items()):
            summary.append(f"  {gate}: {error:.4f}")
        if self.qubit_errors:
            summary.append("Qubit-Specific Errors:")
            for qubit, errors in sorted(self.qubit_errors.items()):
                summary.append(f"  Qubit {qubit}:")
                for gate, error in sorted(errors.items()):
                    summary.append(f"    {gate}: {error:.4f}")
        return "\n".join(summary)

class QuantumNoiseModelFactory:
    """Factory class for creating and caching noise models."""

    @staticmethod
    @lru_cache(maxsize=32)  # Cache up to 32 different noise models
    def create_noise_model(config: QuantumNoiseConfig) -> NoiseModel:
        """
        Create a noise model with the specified configuration.

        Args:
            config: QuantumNoiseConfig object specifying error rates

        Returns:
            NoiseModel: Configured noise model
        """
        noise_model = NoiseModel()

        # Add gate-specific errors
        for gate_name, error_prob in config.gate_errors.items():
            if gate_name in QuantumNoiseConfig.SINGLE_QUBIT_GATES:
                error = depolarizing_error(error_prob, 1)
                noise_model.add_all_qubit_quantum_error(error, gate_name)
                logger.debug(f"Added global error for {gate_name} gate: {error_prob:.4f}")
            elif gate_name in QuantumNoiseConfig.TWO_QUBIT_GATES:
                error = depolarizing_error(error_prob, 2)
                noise_model.add_all_qubit_quantum_error(error, gate_name)
                logger.debug(f"Added global error for {gate_name} gate: {error_prob:.4f}")

        # Add qubit-specific errors if provided
        for qubit_idx, qubit_errors in config.qubit_errors.items():
            for gate_name, error_prob in qubit_errors.items():
                if gate_name in QuantumNoiseConfig.SINGLE_QUBIT_GATES:
                    error = depolarizing_error(error_prob, 1)
                    noise_model.add_quantum_error(error, gate_name, [qubit_idx])
                    logger.debug(f"Added qubit-specific error for {gate_name} gate on qubit {qubit_idx}: {error_prob:.4f}")
                elif gate_name in QuantumNoiseConfig.TWO_QUBIT_GATES:
                    error = depolarizing_error(error_prob, 2)
                    target_qubit = (qubit_idx + 1) % 4  # Assuming 4 qubits
                    noise_model.add_quantum_error(error, gate_name, [qubit_idx, target_qubit])
                    logger.debug(f"Added qubit-specific error for {gate_name} gate on qubits {qubit_idx}-{target_qubit}: {error_prob:.4f}")

        logger.info(f"Created noise model with {len(config.gate_errors)} global gates and "
                   f"{len(config.qubit_errors) if config.qubit_errors else 0} qubit-specific errors")
        return noise_model
