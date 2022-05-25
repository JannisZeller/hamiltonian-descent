# %% Imports
#-------------------------------------------------------------------------------

import tensorflow as tf
import tensorflow.keras as keras

#-------------------------------------------------------------------------------




# %% Implementation
#-------------------------------------------------------------------------------

class Hagrad(keras.optimizers.Optimizer):
    def __init__(self, 
        epsilon: float=1., 
        gamma: float=2., 
        name: str="hagrad", 
        kinetic_energy: str="relativistic", 
        **kwargs
    ): 
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self.kinetic_energy = kwargs.get("kinetic_energy", kinetic_energy)
        self._set_hyper("epsilon", kwargs.get("lr", epsilon)) 
        self._set_hyper("gamma", kwargs.get("gamma", gamma))
        self._set_hyper("delta", 1. / (1. + kwargs.get("lr", epsilon)*kwargs.get("gamma", gamma) ))
    
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

        if self.kinetic_energy == "classical":
            var.assign_add(epsilon * p_var)

        if self.kinetic_energy == "relativistic":
            var.assign_add(epsilon * p_var / tf.math.sqrt(tf.math.square(tf.norm(p_var)) + 1.))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "epsilon": self._serialize_hyperparameter("epsilon"),
            "gamma":   self._serialize_hyperparameter("gamma"),
            "delta":   self._serialize_hyperparameter("delta"),
            "kinetic_energy": self.kinetic_energy,
        }

#-------------------------------------------------------------------------------




# %%
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Hagrad-Optimizer implementation source file ")

#-------------------------------------------------------------------------------
