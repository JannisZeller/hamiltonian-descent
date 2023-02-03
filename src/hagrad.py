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
import tensorflow.keras as keras
# from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from .kinetic_energy_gradients import KineticEnergyGradients

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops

# ------------------------------------------------------------------------------




# %% Implementation
# Implementation of custom tf/tf.keras optimizer inspired by 
# https://cloudxlab.com/blog/writing-custom-optimizer-in-tensorflow-and-keras/
# ------------------------------------------------------------------------------
class Hagrad(keras.optimizers.Optimizer):  # (optimizer_v2.OptimizerV2):
    r"""Optimizer that implements the Hamiltonian Gradient Descent algorithm.
    This optimizer 

    Usage Example:
        >>> opt = tf.keras.optimizers.Hagrad()
        >>> var1 = tf.Variable(10.0)
        >>> loss = lambda: (var1 ** 2) / 2.0
        >>> step_count = opt.minimize(loss, [var1]).numpy()
        >>> "{:.1f}".format(var1.numpy())
        9.4

    Reference:
        - [Maddison et al. (2018)](https://arxiv.org/abs/1809.05042).
    """
    def __init__(self, 
        epsilon: float=1., 
        gamma:   float=10., 
        name:    str="hagrad", 
        kinetic_energy_gradient: Callable=None,
        p0_mean: float=1.,
        p0_std:  float=2.,
        **kwargs): 
        """Returns Hagrad - a keras optimizer. Tested with image and text data. 
        As reliable as SGD but with faster convergence. Refer to the README for
        the usage of parameters.
        """
        super().__init__(name, **kwargs)
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
    

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _create_slots(self, var_list):
        var_dtype = var_list[0].dtype.base_dtype

        ## Initializing kinetic energy for t=0.
        p0_mean = self._get_hyper("p0_mean", var_dtype)
        p0_std  = self._get_hyper("p0_std", var_dtype)
        for var in var_list:
            self.add_slot(
                var, "hamilton_momentum",
                init_ops_v2.random_normal_initializer(mean=p0_mean, stddev=p0_std))
    

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        epsilon     = self._get_hyper("epsilon", var_dtype)
        delta       = self._get_hyper("delta", var_dtype)
        eps_delta   = self._get_hyper("eps_delta", var_dtype)

        p = self.get_slot(var, "hamilton_momentum")
        p = state_ops.assign(p, delta * p + eps_delta * grad)
        var_t = epsilon * self.kinetic_energy_gradient(p)
        return state_ops.assign_add(var, var_t).op


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        epsilon     = self._get_hyper("epsilon", var_dtype)
        delta       = self._get_hyper("delta", var_dtype)
        eps_delta   = self._get_hyper("eps_delta", var_dtype)

        p = self.get_slot(var, "hamilton_momentum")

        p_eps_delta_grad = eps_delta*grad
        p_t = state_ops.assign(p, delta*p)
        with ops.control_dependencies([p_t]):
            p_t = self._resource_scatter_add(
                p_t, 
                indices,
                p_eps_delta_grad)

        var_t = state_ops.assign_add(
            var, epsilon * self.kinetic_energy_gradient(p_t))

        return control_flow_ops.group(*[var_t, p_t])


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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

    ## Setup
    import tensorflow.keras as keras
    import numpy as np

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