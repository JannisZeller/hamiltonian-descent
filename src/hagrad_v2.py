# Copyright 2022 Jannis Zeller. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hagrad optimizer implementation."""
# pylint: disable=g-classes-have-attributes

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops_v2
# from tensorflow.python.util.tf_export import keras_export



# @keras_export('keras.optimizers.Hagrad')
class Hagrad(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Hamiltonian Gradient Descent algorithm.
  This optimizer 

  Usage Example:
    >>> opt = tf.keras.optimizers.Hagrad()
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2) / 2.0
    >>> step_count = opt.minimize(loss, [var1]).numpy()
    >>> "{:.1f}".format(var1.numpy())
    9.8

  Reference:
    - [Maddison et al. (2018)](https://arxiv.org/abs/1809.05042).
  """

  _HAS_AGGREGATE_GRAD = False

  def __init__(self, 
               epsilon=1., 
               gamma=10., 
               name="hagrad", 
               # kinetic_energy_gradient=None,
               p0_mean=1.,
               p0_std=2.,
               **kwargs): 
    super().__init__(name, **kwargs)
    # if kinetic_energy_gradient == None:
    #     self.kinetic_energy_gradient = KineticEnergyGradients.relativistic()
    # else:
    #     self.kinetic_energy_gradient = kinetic_energy_gradient
    self._set_hyper("epsilon", epsilon) 
    self._set_hyper("gamma",   gamma)
    self._set_hyper("p0_mean", p0_mean) 
    self._set_hyper("p0_std",  p0_std)
    delta =  1. / (1. + epsilon * gamma )
    self._set_hyper("delta", delta)
    eps_delta = -1. * epsilon * delta
    self._set_hyper("eps_delta", eps_delta)

  def _create_slots(self, var_list):
    var_dtype = var_list[0].dtype.base_dtype
    ## Initializing first momentums.
    p0_mean = self._get_hyper("p0_mean", var_dtype)
    p0_std  = self._get_hyper("p0_std", var_dtype)
    for var in var_list:
        self.add_slot(
            var, "hamilton_momentum",
            init_ops_v2.random_normal_initializer(mean=p0_mean, stddev=p0_std)) 
    
  # @tf.function
  def _resource_apply_dense(self, grad, var):
      var_dtype = var.dtype.base_dtype
      epsilon     = self._get_hyper("epsilon", var_dtype)
      delta       = self._get_hyper("delta", var_dtype)
      eps_delta   = self._get_hyper("eps_delta", var_dtype)

      p = self.get_slot(var, "hamilton_momentum")

      p.assign(delta * p + eps_delta * grad)
      var.assign_add(epsilon * self.kinetic_energy_gradient(p))

  # @tf.function
  def _resource_apply_sparse(self, grad, var, indices):
      var_dtype = var.dtype.base_dtype
      epsilon     = self._get_hyper("epsilon", var_dtype)
      delta       = self._get_hyper("delta", var_dtype)
      eps_delta   = self._get_hyper("eps_delta", var_dtype)

      p = self.get_slot(var, "hamilton_momentum")

      p_ = math_ops.tensor_scatter_nd_add(
          delta*p, 
          array_ops.expand_dims(indices, -1), 
          eps_delta*grad)

      p.assign(p_)
      var.assign_add(epsilon * self.kinetic_energy_gradient(p))

  def get_config(self):
    config = super().get_config()
    config.update({
      'epsilon': self._serialize_hyperparameter("epsilon"),
      'gamma':   self._serialize_hyperparameter("gamma"),
      'delta':   self._serialize_hyperparameter("delta"),
      # 'kinetic_energy_gradient': self.kinetic_energy_gradient.__doc__
    })
    return config
