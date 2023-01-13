##### HaGraD #####
# - - - - - - - - 
# Source file for the HaGraD optimizer.
# Import as `import Hagrad from hagrad`.

# To run the test case, execute something like 
#   `python -m src.kinetic_energy_gradients` 
# in the terminal.
# ------------------------------------------------------------------------------



# %% Imports
# ------------------------------------------------------------------------------

import tensorflow as tf
from typing import Callable

# ------------------------------------------------------------------------------




# %% Implementation
# ------------------------------------------------------------------------------
class KineticEnergyGradients():
    """Provides several functions which serve as kinetic energy gradients in 
    Hagrad.
    """
    @staticmethod
    def classical() -> Callable:
        def classical_func(p: tf.Tensor) -> tf.Tensor:
            """Classical kinetic energy ||p||^2/2 with gradient p."""
            return p
        
        return classical_func

    @staticmethod
    def relativistic() -> Callable:
        def relativistic_func(p: tf.Tensor) -> tf.Tensor:
            """Relativistic kinetic energy sqrt( ||p||^2 + 1 )-1 with gradient p/sqrt( ||p||^2 + 1 )"""
            return( p / tf.math.sqrt(tf.math.square(tf.norm(p)) + 1.) )
        
        return relativistic_func

    @staticmethod
    def power(a: float = 2., A: float = 1.) -> Callable:
        a = float(a)
        A = float(A)
        
        def power_func(p: tf.Tensor) -> tf.Tensor:
            # docstring appended afterwards
            p_norm = tf.norm(p)
            return( p * p_norm**(a-2.) * (p_norm**a + 1.)**(A/a - 1.) )
        power_func.__doc__ = f"Power kinetic energy (1/{A}) * ( ||p||^{a} + 1 )^({A}/{a}) - (1/{A}) with gradient p * ||p||^({a}-2) * ( ||p||^{a} + 1 )^({A}/{a}-1)"
        
        return power_func

# ------------------------------------------------------------------------------




# %% Main (Test Case)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running kinetic_energy_gradients test case.")
    print("-------------------------------------------")

    import numpy as np
    p = np.random.normal(size=(10, 10))
    
    classical_test = KineticEnergyGradients.classical()
    _ = classical_test(p)
    relativistic_test = KineticEnergyGradients.relativistic()
    _ = relativistic_test(p)
    power_test = KineticEnergyGradients.power()
    _ = power_test(p)

    print("Tests completed successfully.")
    print("-----------------------------")

# ------------------------------------------------------------------------------