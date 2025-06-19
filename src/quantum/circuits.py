import numpy as np
import torch
import time
import logging
from typing import Optional, Dict, Any
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, circuit_drawer

from .noise import QuantumNoiseConfig, QuantumNoiseModelFactory

# Set up logging
logger = logging.getLogger('quantum_mri')

class QuantumCircuitError(Exception):
    """Raised when there's an error in quantum circuit creation or execution."""
    pass

class BaseQuantumCircuit:
    """Base class for quantum circuits with shared noise handling."""

    # Class-level noise model factory
    _noise_factory = QuantumNoiseModelFactory()

    def __init__(self,
                 patch: np.ndarray,
                 noise_config: Optional[QuantumNoiseConfig] = None,
                 use_gpu: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initialize the quantum circuit.

        Args:
            patch: Input patch to encode
            noise_config: Optional noise configuration
            use_gpu: Whether to use GPU
            device: Torch device to use
        """
        if patch.shape != (2, 2):
            raise QuantumCircuitError(f"Expected 2x2 patch, got shape {patch.shape}")

        self.patch = patch
        self.use_gpu = use_gpu
        self.device = device

        # Use default noise config if none provided
        if noise_config is None:
            noise_config = QuantumNoiseConfig({
                'h': 0.001,    # Hadamard gates
                'ry': 0.002,   # Rotation-Y gates
                'rz': 0.002,   # Rotation-Z gates
                'cx': 0.01     # CNOT gates
            })

        self.noise_config = noise_config
        self.noise_model = self._noise_factory.create_noise_model(noise_config)

        # Circuit state tracking
        self._circuit_hash = None
        self._qasm_string = None
        self._last_execution_time = None
        self._last_shots = None
        self._last_counts = None

        # Create circuit
        self.circuit = None
        self._create_circuit()

        logger.debug(f"Created circuit with noise config: {noise_config.gate_errors}")

    def _create_circuit(self):
        """Abstract method to be implemented by derived classes."""
        raise NotImplementedError("Derived classes must implement _create_circuit")

    def _update_circuit_state(self):
        """Update circuit state tracking information."""
        if self.circuit is not None:
            try:
                # Use QASM 2.0 for circuit representation
                from qiskit.qasm2 import dumps
                self._qasm_string = dumps(self.circuit)
                # Create a hash based on circuit properties
                self._circuit_hash = hash((
                    self.circuit.depth(),
                    self.circuit.num_qubits,
                    self._qasm_string
                ))
            except Exception as e:
                logger.warning(f"Failed to update circuit state: {str(e)}")
                # Fallback to basic circuit properties
                self._circuit_hash = hash((
                    self.circuit.depth(),
                    self.circuit.num_qubits,
                    str(self.circuit.data)
                ))
                self._qasm_string = str(self.circuit)

    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the current circuit state."""
        return {
            'circuit_hash': self._circuit_hash,
            'qasm_string': self._qasm_string,
            'last_execution_time': self._last_execution_time,
            'last_shots': self._last_shots,
            'last_counts': self._last_counts,
            'noise_config': self.noise_config.get_summary()
        }

    def execute(self, shots: int = 1000) -> np.ndarray:
        """
        Execute the quantum circuit with the configured noise model.

        Args:
            shots: Number of shots to run the circuit

        Returns:
            np.ndarray: Z-expectation values for each qubit
        """
        try:
            start_time = time.time()

            # Update circuit state
            self._update_circuit_state()

            # Execute circuit
            simulator = AerSimulator()
            job = simulator.run(self.circuit, shots=shots, noise_model=self.noise_model)
            result = job.result()
            counts = result.get_counts()

            # Calculate Z-expectations
            z_expectations = np.zeros(4)
            total_shots = sum(counts.values())

            for state, count in counts.items():
                state_array = np.array([int(bit) for bit in state])
                z_expectations += (1 - 2 * state_array) * (count / total_shots)

            # Update execution state
            self._last_execution_time = time.time() - start_time
            self._last_shots = shots
            self._last_counts = counts

            logger.debug(f"Circuit executed in {self._last_execution_time:.3f}s with {shots} shots")
            return z_expectations

        except Exception as e:
            logger.error(f"Circuit execution failed: {str(e)}")
            raise QuantumCircuitError(f"Circuit execution failed: {str(e)}")

class AmplitudeQuantumCircuit(BaseQuantumCircuit):
    """Quantum circuit with amplitude encoding."""

    def _create_circuit(self):
        """Create the amplitude encoding circuit."""
        try:
            qr = QuantumRegister(4, 'q')
            cr = ClassicalRegister(4, 'c')
            self.circuit = QuantumCircuit(qr, cr)

            patch_flat = self.patch.flatten()
            norm = np.linalg.norm(patch_flat)
            if norm == 0:
                raise QuantumCircuitError("Patch contains only zeros")
            normalized_patch = patch_flat / norm

            # Apply Hadamard gates
            for i in range(4):
                self.circuit.h(i)

            # Apply rotation gates for amplitude encoding
            for i, value in enumerate(normalized_patch):
                if i < 3:
                    self.circuit.ry(2 * np.arcsin(value), i)
                    self.circuit.rz(2 * np.arccos(value), i)

            # Apply CNOT gates
            for i in range(4):
                self.circuit.cx(i, (i + 1) % 4)

            self.circuit.measure(qr, cr)

        except Exception as e:
            logger.error(f"Failed to create amplitude encoding circuit: {str(e)}")
            raise QuantumCircuitError(f"Circuit creation failed: {str(e)}")

class AngleQuantumCircuit(BaseQuantumCircuit):
    """Quantum circuit with angle encoding."""

    def _create_circuit(self):
        """Create the angle encoding circuit."""
        try:
            qr = QuantumRegister(4, 'q')
            cr = ClassicalRegister(4, 'c')
            self.circuit = QuantumCircuit(qr, cr)

            # Apply rotation gates for angle encoding
            for i, value in enumerate(self.patch.flatten()):
                self.circuit.ry(value, i)
                self.circuit.rz(value * value, i)

            # Apply CNOT gates
            for i in range(4):
                self.circuit.cx(i, (i + 1) % 4)

            self.circuit.measure(qr, cr)

        except Exception as e:
            logger.error(f"Failed to create angle encoding circuit: {str(e)}")
            raise QuantumCircuitError(f"Circuit creation failed: {str(e)}")
