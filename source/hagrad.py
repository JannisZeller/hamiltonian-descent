# %% Imports
#-------------------------------------------------------------------------------

import warnings
from typing import Callable

import tensorflow as tf
import tensorflow.keras as keras

#-------------------------------------------------------------------------------




# %% Implementation
#-------------------------------------------------------------------------------

class KineticEnergyGradients():
    """This class provides several functions which serve as kinetic energy gradients in Hagrad.
    """

    def __init__(self, 
        power_a: float=2.,
        power_A: float=1.
    ):
        self.power_a = power_a
        self.power_A = power_A

    @tf.function
    def classical(p_var: tf.Tensor) -> tf.Tensor:
        """Classical kinetic energy ||p||^2/2 with gradient p."""
        return(p_var)

    @tf.function
    def relativistic(p_var: tf.Tensor) -> tf.Tensor:
        """Relativistic kinetic energy sqrt( ||p||^2 + 1 )-1 with gradient p/sqrt( ||p||^2 + 1 )"""
        return(p_var / tf.math.sqrt(tf.math.square(tf.norm(p_var)) + 1.))

    @tf.function
    def power(self, p_var: tf.Tensor) -> tf.Tensor:
        """Power kinetic energy (1/A) * ( ||p||^a + 1 )^(A/a) - (1/A) with gradient p * ||p||^(a-2) * ( ||p||^a + 1 )^(A/a-1)"""
        a = self.power_a
        A = self.power_A
        p_norm = tf.norm(p_var)
        return(p_var*p_norm**(a-2.) * (p_norm**a + 1.)**(A/a-1.))

    @tf.function
    def power_1norm(self, p_var: tf.Tensor) -> tf.Tensor:
        """Power kinetic energy (1/A) * ( ||p||^a + 1 )^(A/a) - (1/A) with gradient p * ||p||^(a-2) * ( ||p||^a + 1 )^(A/a-1) where ||p|| is the 1-norm."""
        a = self.power_a
        A = self.power_A
        p_norm = tf.norm(p_var, ord=1)
        return(tf.math.sign(p_var)*p_norm**(a-1.) * (p_norm**a + 1.)**(A/a-1.))


class Hagrad(keras.optimizers.Optimizer):

    def __init__(self, 
        epsilon: float=1., 
        gamma:   float=10., 
        name:    str="hagrad", 
        kinetic_energy: str="relativistic", # DEPRECATED

        # TODO
        kinetic_energy_gradient: Callable[[tf.Tensor], tf.Tensor]=KineticEnergyGradients.relativistic,

        **kwargs
    ): 
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self.kinetic_energy = kwargs.get("kinetic_energy", kinetic_energy)

        # TODO
        self.kinetic_energy_gradient = kwargs.get("kinetic_energy_gradient", kinetic_energy_gradient)

        self._set_hyper("epsilon", kwargs.get("lr", epsilon)) 
        self._set_hyper("gamma", kwargs.get("gamma", gamma))
        self._set_hyper("delta", 1. / (1. + kwargs.get("lr", epsilon)*kwargs.get("gamma", gamma) ))

        if self.kinetic_energy_gradient.__name__ == "power_1norm":
            warnings.warn("The power_1norm kinetic energy gradient typically leads to divergence!")
    
    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "hamilton_momentum") 

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        p_var = self.get_slot(var, "hamilton_momentum")
        epsilon = self._get_hyper("epsilon", var_dtype)
        delta   = self._get_hyper("delta", var_dtype)
        p_var.assign(delta * p_var - epsilon * delta * grad)

        # TODO
        var.assign_add(epsilon * self.kinetic_energy_gradient(p_var))


        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        ## -> It seems like some (small amount of) computation time can be saved, if one does 
        #  not use conditions or a property to store different kinetic energy gradients.
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 


        # var.assign_add(epsilon * p_var) 

        ## DEPRECATED
        # if self.kinetic_energy == "classical":
        #     var.assign_add(epsilon * p_var)

        # if self.kinetic_energy == "relativistic":
        #     var.assign_add(epsilon * p_var / tf.math.sqrt(tf.math.square(tf.norm(p_var)) + 1.))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'epsilon': self._serialize_hyperparameter("epsilon"),
            'gamma':   self._serialize_hyperparameter("gamma"),
            'delta':   self._serialize_hyperparameter("delta"),
            'kinetic_energy': "deprecated", ## DEPRECATED
            'kinetic_energy_gradient': self.kinetic_energy_gradient.__doc__
        }

#-------------------------------------------------------------------------------




# %%
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Hagrad-Optimizer implementation source file.")

#-------------------------------------------------------------------------------
