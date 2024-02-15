from typing import Union, Optional, Tuple, List, Any
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Isometry, Initialize
from scipy.stats import expon
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

def _check_dimensions_match(num_qubits, lambd, bounds):
    num_qubits = [num_qubits] if not isinstance(num_qubits, (list, np.ndarray)) else num_qubits
    dim = len(num_qubits)

    if lambd is not None:
        lambd = [lambd] if not isinstance(lambd, (list, np.ndarray)) else lambd
        if len(lambd) != dim:
            raise ValueError(
                f"Dimension of lambd ({len(lambd)}) does not match the dimension of the "
                f"random variable specified by the number of qubits ({dim})"
            )

    if bounds is not None:
        bounds = [bounds] if not isinstance(bounds[0], tuple) else bounds
        if len(bounds) != dim:
            raise ValueError(
                f"Dimension of bounds ({len(bounds)}) does not match the dimension of the "
                f"random variable specified by the number of qubits ({dim})"
            )

def _check_bounds_valid(bounds):
    if bounds is None:
        return

    bounds = [bounds] if not isinstance(bounds[0], tuple) else bounds

    for i, bound in enumerate(bounds):
        if not bound[1] - bound[0] > 0:
            raise ValueError(
                f"Dimension {i} of the bounds are invalid, must be a non-empty "
                "interval where the lower bounds is smaller than the upper bound."
            )

class ExponentialDistribution(QuantumCircuit):
    def __init__(
        self,
        num_qubits: Union[int, List[int]],
        lambd: Optional[Union[float, List[float]]] = None,
        bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        upto_diag: bool = False,
        name: str = "P(X)",
    ) -> None:
        r"""
        Args:
            num_qubits: The number of qubits used to discretize the random variable. For a 1d
                random variable, ``num_qubits`` is an integer, for multiple dimensions a list
                of integers indicating the number of qubits to use in each dimension.
            lambd: The rate parameter :math:`\lambda`, which is the reciprocal of the expected
                value of the distribution. Can be either a float for a 1d random variable or a
                list of floats for a higher dimensional random variable. Defaults to 1.
            bounds: The truncation bounds of the distribution as tuples. For multiple dimensions,
                ``bounds`` is a list of tuples ``[(low0, high0), (low1, high1), ...]``.
                If ``None``, the bounds are set to ``(-1, 1)`` for each dimension.
            upto_diag: If True, load the square root of the probabilities up to multiplication
                with a diagonal for a more efficient circuit.
            name: The name of the circuit.
        """
        _check_dimensions_match(num_qubits, lambd, bounds)
        _check_bounds_valid(bounds)

        # set default arguments
        dim = 1 if isinstance(num_qubits, int) else len(num_qubits)
        if lambd is None:
            lambd = 1 if dim == 1 else [1] * dim

        if bounds is None:
            bounds = (-1, 1) if dim == 1 else [(-1, 1)] * dim

        if isinstance(num_qubits, int):  # univariate case
            inner = QuantumCircuit(num_qubits, name=name)

            x = np.linspace(bounds[0], bounds[1], num=2**num_qubits)  # type: Any
        else:  # multivariate case
            inner = QuantumCircuit(sum(num_qubits), name=name)

            # compute the evaluation points using numpy.meshgrid
            # indexing 'ij' yields the "column-based" indexing
            meshgrid = np.meshgrid(
                *[
                    np.linspace(bound[0], bound[1], num=2 ** num_qubits[i])  # type: ignore
                    for i, bound in enumerate(bounds)
                ],
                indexing="ij",
            )
            # flatten into a list of points
            x = list(zip(*[grid.flatten() for grid in meshgrid]))

        # compute the normalized, truncated probabilities
        probabilities = expon.pdf(x, scale=1/np.array(lambd))
        print(probabilities)
        normalized_probabilities = probabilities / np.sum(probabilities)

        # store the values, probabilities and bounds to make them user accessible
        self._values = x
        self._probabilities = normalized_probabilities
        self._bounds = bounds

        super().__init__(*inner.qregs, name=name)

        # use default the isometry (or initialize w/o resets) algorithm to construct the circuit
        if upto_diag:
            inner.append(Isometry(np.sqrt(normalized_probabilities), 0, 0), inner.qubits)
            self.append(inner.to_instruction(), inner.qubits)  # Isometry is not a Gate
        else:
            initialize = Initialize(np.sqrt(normalized_probabilities))
            circuit = initialize.gates_to_uncompute().inverse()
            inner.compose(circuit, inplace=True)
            self.append(inner.to_gate(), inner.qubits)

    @property
    def values(self) -> np.ndarray:
        """Return the discretized points of the random variable."""
        return self._values

    @property
    def probabilities(self) -> np.ndarray:
        """Return the sampling probabilities for the values."""
        return self._probabilities

    @property
    def bounds(self) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Return the bounds of the probability distribution."""
        return self._bounds


# Test the values property
exp_dist = ExponentialDistribution(5, lambd=1.5, bounds=(0, 2))

print("Values:")
print(exp_dist.values)

# Test the probabilities property
print("\nProbabilities:")
print(exp_dist.probabilities)

# Test the bounds property
print("\nBounds:")
print(exp_dist.bounds)


# Visualize the histogram of the probabilities
plt.bar(exp_dist.values, exp_dist.probabilities)
plt.xlabel('Values')
plt.ylabel('Probabilities')
plt.title('Histogram of Exponential Distribution')
plt.show()

# Plot the quantum circuit
exp_dist.decompose().draw(output='mpl',filename="face.png")