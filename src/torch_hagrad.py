##### HaGraD #####
# - - - - - - - - 
# Source file for the HaGraD optimizer.
# Import as `import Hagrad from torch_hagrad`.
# 
# ------------------------------------------------------------------------------



# %% Imports
# ------------------------------------------------------------------------------

import torch
from torch import Tensor
from torch.optim import Optimizer

# ------------------------------------------------------------------------------




# %% Implementation
# ------------------------------------------------------------------------------
class Hagrad(Optimizer):
    r"""Implements hamiltonian descent.
    """

    def __init__(self, 
        params, 
        kinetic_energy: str="relativistic",
        epsilon=1., 
        gamma=10., 
        p_mean=0., 
        p_std=1.):
        if epsilon < 0.0:
            raise ValueError(f"Invalid negative learning rate: {epsilon}")
        if kinetic_energy not in ["relativistic", "classical"]:
            raise ValueError(f"Invalid kinetic_energy: {kinetic_energy}. Choose from [\"relativistic\", \"classical\"]")

        if not isinstance(params, list):
            params = list(params)
        self.params = params

        ## Initializing Momenta
        momenta = []
        for param in params:
            momenta.append(torch.normal(
                p_mean*torch.ones_like(param), 
                p_std *torch.ones_like(param)).requires_grad_(True))
        self.momenta = momenta

        ## Kinetic Energy
        if kinetic_energy == "relativistic":
            self.dT = lambda p: p / torch.sqrt(torch.square(torch.linalg.norm(p)) + 1.)
        elif kinetic_energy == "classical":
            self.dT = lambda p: p

        ## Storing hypers
        self.eps = epsilon
        self.gamma = gamma
        self.delta = (1. + gamma*epsilon)**(-1.)
        self.eps_delta = epsilon*self.delta


    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.params_with_grad = []
            
        for x, p in zip(self.params, self.momenta):
            x_grad = x.grad
            if x_grad is not None:
                self.params_with_grad.append(x)
                p.data.mul_(self.delta).add_(-self.eps_delta*x_grad)
                x.data.add_(self.eps*self.dT(p))

        return loss


    def zero_grad(self):
        if hasattr(self, "params_with_grad"):
            for x in self.params_with_grad:
                x.grad.zero_()
                
# ------------------------------------------------------------------------------
