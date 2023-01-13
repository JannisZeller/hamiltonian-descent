##### HaGraD #####
# - - - - - - - - 
# Source file for the HaGraD optimizer.
# Import as `import Hagrad from hagrad`.
# 
# To run the test case, execute something like 
#   `python -m src.hagrad` 
# in the terminal.
# ------------------------------------------------------------------------------



# %% Imports
# ------------------------------------------------------------------------------

from typing import Callable
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from .kinetic_energy_gradients import KineticEnergyGradients

# ------------------------------------------------------------------------------




# %% Implementation
# Implementation of custom tf/tf.keras optimizer inspired by 
# https://cloudxlab.com/blog/writing-custom-optimizer-in-tensorflow-and-keras/
# ------------------------------------------------------------------------------
class Hagrad(keras.optimizers.Optimizer):

    def __init__(self, 
        epsilon: float=1., 
        gamma:   float=10., 
        name:    str="hagrad", 
        kinetic_energy_gradient: Callable=None,
        p0_mean: float=1.,
        p0_std:  float=2.,
        **kwargs): 

        ## Call super().__init__() and...
        super().__init__(name, **kwargs)
        
        ## ...use _set_hyper() to store hyperparameters
        if kinetic_energy_gradient == None:
            self.kinetic_energy_gradient = KineticEnergyGradients.relativistic()
        else:
            self.kinetic_energy_gradient = kinetic_energy_gradient
        self._set_hyper("epsilon", epsilon) 
        self._set_hyper("gamma",   gamma)
        self._set_hyper("p0_mean", p0_mean) 
        self._set_hyper("p0_std",  p0_std)
        delta =  1. / (1. + epsilon * gamma )
        self._set_hyper("delta", delta)
        eps_delta = -1. * epsilon * delta
        self._set_hyper("eps_delta", eps_delta)
    

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        var_dtype = var_list[0].dtype.base_dtype

        ## Initializing kinetic energy for t=0.
        p0_mean = self._get_hyper("p0_mean", var_dtype)
        p0_std  = self._get_hyper("p0_std", var_dtype)
        for var in var_list:
            self.add_slot(
                var, "hamilton_momentum",
                tf.random_normal_initializer(mean=p0_mean, stddev=p0_std)) 
    

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        epsilon     = self._get_hyper("epsilon", var_dtype)
        delta       = self._get_hyper("delta", var_dtype)
        eps_delta   = self._get_hyper("eps_delta", var_dtype)

        p = self.get_slot(var, "hamilton_momentum")

        p.assign(delta * p + eps_delta * grad)
        var.assign_add(epsilon * self.kinetic_energy_gradient(p))


    @tf.function
    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        epsilon     = self._get_hyper("epsilon", var_dtype)
        delta       = self._get_hyper("delta", var_dtype)
        eps_delta   = self._get_hyper("eps_delta", var_dtype)

        p = self.get_slot(var, "hamilton_momentum")

        p_ = tf.tensor_scatter_nd_add(
            delta*p, 
            tf.expand_dims(indices, -1), 
            eps_delta*grad)

        p.assign(p_)
        var.assign_add(epsilon * self.kinetic_energy_gradient(p))


    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'epsilon': self._serialize_hyperparameter("epsilon"),
            'gamma':   self._serialize_hyperparameter("gamma"),
            'delta':   self._serialize_hyperparameter("delta"),
            'kinetic_energy_gradient': self.kinetic_energy_gradient.__doc__
        }

# ------------------------------------------------------------------------------




# %% Main (Test Case)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running HaGraD test case.")
    print("-------------------------")

    ## Define Optimizer
    hagrad = Hagrad(
        p0_mean=0.001,
        kinetic_energy_gradient=KineticEnergyGradients.relativistic())
    hagrad.get_config()

    ## Generating Data (checkerboard)
    X = 2 * (np.random.rand(1000, 2) - 0.5)
    y = np.array(X[:, 0] * X[:, 1] > 0, np.int32)

    ## Define Model
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    ## Compile Model
    model.compile(
        loss=keras.losses.binary_crossentropy, 
        optimizer=hagrad, 
        metrics=["accuracy"])
    
    model.fit(X, y, epochs=10, batch_size=32)

    print("Tests completed successfully.")
    print("-----------------------------")

# ------------------------------------------------------------------------------