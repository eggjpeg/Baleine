from typing import Union, Optional, Tuple, List, Any
import numpy as np
from qiskit import Aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.library import Isometry, Initialize
from scipy.stats import expon
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import math

def _check_bounds_valid(bounds):
    if bounds is None:
        return




class ExponentialDistribution(QuantumCircuit):
    def __init__(
        self,
        num_qubits: int,
        lambd: Optional[float] = None,
        bounds: Optional[Union[Tuple[float, float]]] = None,
        xnot: Optional[float] = None,
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
            xnot: The value that we choose to begin plotting at.
            upto_diag: If True, load the square root of the probabilities up to multiplication
                with a diagonal for a more efficient circuit.
            name: The name of the circuit.
        """
        _check_bounds_valid(bounds)

        # set default arguments
        if lambd is None:
            lambd = 1 
        if lambd <= 0:
            raise ValueError("Lambd must be positive")
        if bounds is None:
            bounds = (0,-np.log(0.01) / lambd)
        if xnot is None:
            xnot = (bounds[1] + bounds[0]) / 2
        if xnot > bounds[1] or xnot < bounds[0]:
            raise ValueError("xnot must be within the bounds!")
        

        inner = QuantumCircuit(num_qubits, name=name)

        #xp = np.array(np.linspace(xnot, bounds[1], num=int((2**num_qubits) / 2)))  # type: Any
        #xn = np.negative(np.flip(xp))
        #x = np.concatenate([xn,xp])


        x = np.array(np.linspace(bounds[0], bounds[1], num=int((2**num_qubits)) ))  # type: Any

        # We get the first and second half of x values before x not and after
        xFirstHalf = x[x <= xnot]
        xSecondHalf = x[x > xnot]

        # We check to see if the first half or second half is larger, depending on xnot
        if(len(xFirstHalf) > len(xSecondHalf)):
            # We calculate the probabilities before xnot, we must flip the first half to do this
            probabilitiesn = lambd * np.exp(-lambd*(np.flip(xFirstHalf) - xnot))
            # We then flip to get the probabilties after xnot, now the size of this array will be too large so we must truncate it after
            probabilitiesp = np.flip(probabilitiesn)
        else:
            # In this case we must flip the probabilities before xnot and truncate them at the point that we want
            probabilitiesp = lambd * np.exp(-lambd*(xSecondHalf - xnot))
            probabilitiesn = np.flip(probabilitiesp[:len(xFirstHalf)])
            
        
        probabilities = np.concatenate([probabilitiesn,probabilitiesp])

        #We truncate here for the first case, the second case is unaffected by this
        probabilities = probabilities[:len(x)]

        # Note:
        # I end up having 2 probabilities at xnot because of the flip, how to fix?

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
    def bounds(self) ->  Optional[Union[Tuple[float, float]]]:
        """Return the bounds of the probability distribution.   """
        return self._bounds


#######################################################################################################################################
    
def get_loss(sigma, bounds) -> float:
    return math.exp(-sigma * abs(bounds[1] - bounds[0]))

exp_dist = ExponentialDistribution(7, lambd=2, bounds= (-4,4), xnot=2)

leak = QuantumCircuit(1)
sigma = 1
loss = get_loss(sigma,exp_dist.bounds)
print(loss)


print("Values:")
print(exp_dist.values)

# Test the probabilities property
print("\nProbabilities:")
print(exp_dist.probabilities)

# Test the bounds property
print("\nBounds:")
print(exp_dist.bounds)


# Visualize the histogram of the probabilities
plt.plot(exp_dist.values, exp_dist.probabilities)
plt.xlabel('Values')
plt.ylabel('Probabilities')
plt.title('Histogram of Exponential Distribution')
plt.show()

# Plot the quantum circuit
exp_dist.decompose(reps=8).draw(output='mpl',filename="face.png")