import tensorflow as tf
from typing import Callable

class KineticEnergyGradients():
    """
    Provides several functions which serve as kinetic energy gradients in Hagrad.
    """
    @staticmethod
    def classical() -> Callable:
        @tf.function
        def classical_func(p_var: tf.Tensor) -> tf.Tensor:
            """Classical kinetic energy ||p||^2/2 with gradient p."""
            return(p_var)
        
        return classical_func

    @staticmethod
    def relativistic() -> Callable:       
        @tf.function
        def relativistic_func(p_var: tf.Tensor) -> tf.Tensor:
            """Relativistic kinetic energy sqrt( ||p||^2 + 1 )-1 with gradient p/sqrt( ||p||^2 + 1 )"""
            return(p_var / tf.math.sqrt(tf.math.square(tf.norm(p_var)) + 1.))
        
        return relativistic_func

    @staticmethod
    def power(a: float = 2., A: float = 1.) -> Callable:
        a = float(a)
        A = float(A)
        
        @tf.function
        def power_func(p_var: tf.Tensor) -> tf.Tensor:
            # docstring appended afterwards
            p_norm = tf.norm(p_var)
            return(p_var*p_norm**(a-2.) * (p_norm**a + 1.)**(A/a-1.))
        power_func.__doc__ = f"Power kinetic energy (1/{A}) * ( ||p||^{a} + 1 )^({A}/{a}) - (1/{A}) with gradient p * ||p||^({a}-2) * ( ||p||^{a} + 1 )^({A}/{a}-1)"
        
        return power_func