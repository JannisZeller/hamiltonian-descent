##### HaGraD #####
# - - - - - - - - 
# Source file for the HaGraD optimizer.
# Import as `import Hagrad from torch_hagrad`.
# 
# To run the test case, execute something like 
#   `python -m src.torch_hagrad` 
# in the terminal.
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




# %% Main (Test Case)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running HaGraD test case.")
    print("-------------------------")

    ## Setup
    from time import sleep # To actually see something...
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device.\n")
    
    slp = input("Should I wait between batches (y/n)?") in ["y", ""]
    print()

    def binary_accuracy(preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        return correct.sum() / len(y)

    ## Generating Data (checkerboard)
    X = 2 * (np.random.rand(1000, 2) - 0.5)
    y = np.array(X[:, 0] * X[:, 1] > 0, np.int32)
    X_all = torch.tensor(X, dtype=torch.float32).to(device)
    y_all = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_all, y_all)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    ## Define model
    class DenseNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 1))
        def forward(self, x):
            x = self.flatten(x)
            logs = self.linear_relu_stack(x)
            return logs.squeeze()

    model = DenseNN().to(device)

    ## Define Optimizer and criterion
    hagrad = Hagrad(model.parameters(), "classical")
    criterion = nn.BCEWithLogitsLoss()

    ## Fit the model
    size = len(dataloader.dataset)
    for e in range(10):
        batch_counter = 0
        model.train()
        for X, y in dataloader:
            preds = model(X)
            loss = criterion(preds, y)
            hagrad.zero_grad()
            loss.backward() 
            hagrad.step()
            batch_counter += len(y)
            acc = binary_accuracy(preds, y)
            print(f"\rPassed [{batch_counter:>2d}/{size:>2d}] -- batch loss: {loss.item():.2f} -- batch acc: {acc:.2f}", end="\r")
            if slp:
                sleep(0.03)
        with torch.no_grad():
            model.eval()
            preds = model(X_all)
            loss = criterion(preds, y_all)
            acc = binary_accuracy(preds, y_all)
            print(f"\nEpoch [{e+1:>2d}/10] -- total loss: {loss:.2f} -- total acc: {acc:.2f}\n")

    print("\nTests completed successfully.")
    print("-----------------------------")

# ------------------------------------------------------------------------------