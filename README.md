# HaGraD

This repository presents an implementation of one of the Hamiltonian Descent Methods presented in 

Maddison, C. J., Paulin, D., Teh, Y. W., O'Donoghue, B., & Doucet, A. (2018). *Hamiltonian Descent Methods*. arXiv:[1809.05042](https://arxiv.org/abs/1809.05042)

Creative as we ([Jannis Zeller](https://de.linkedin.com/in/jannis-zeller-12477a221), [David Schischke](https://de.linkedin.com/in/david-schischke-b850ba170)) are, we named the implemented optimizer `HaGraD`.

### Table of Contents 

* [Introduction](#theoretical-background)
* [Theoretical Background](#theoretical-background)
* [Hamiltonian Descent methods](#hamiltonian-descent-methods)
* [Usage of `HaGraD`](#usage-of-hagrad)




## Introduction
---
Many commonly used optimization methods in Machine Learning (*ML*) show linear to sublinear convergence rates. Hamiltonian Descent Methods (*HD*) are a new method that extend linear convergence for a broader class of convex functions (Maddison et al., 2018). We provide a method that implements a specific HD in TensorFlow (`tf.keras`). In the [notebooks](./notebooks/) we compare the method's performance to stochastic gradient descent (*SGD*) and adaptive moment estimation (*Adam*) on two standard visual tasks an on one Natural Language Processing (NLP)-task.




## Theoretical Background 
---

This section provides a brief summary of the HD methods proposed by [Maddison et al. (2018)](https://arxiv.org/abs/1809.05042), adding some context and comparisons to commonly used optimization methods for Neural Networks.  


### Hamiltonian Mechanics

*Refer to any book in theoretical mechanics for the following.* Like other "Hamiltonian-labelled" approaches (e. g. [Hamiltonian Monte Carlo Sampling](https://arxiv.org/abs/2108.12107)) HD methods are rooted in physics. In classical mechanics a system and its propagation is fully specified by space coordinates $x \in \mathbb R^F$ and their corresponding first derivatives $\dot x = v \in \mathbb R^F$. The evolution of the system is then described by the equation of motion (2. Newtonian Axiom): $$m \ddot x = F(t, x, \dot x)\, ,$$
where $m$ is the mass, $t$ is the time and $F$ is the acting force. This is roughly summarized for the sake of simplicity. First the [Lagrangian](https://en.wikipedia.org/wiki/Lagrangian_mechanics) and the the [Hamiltonian](https://en.wikipedia.org/wiki/Hamiltonian_mechanics) reformulations of classical mechanics generalized this. Without going into much details, in these reformulations each "position" variable $q$ corresponds to a "momentum" variable $p$ which are connected through functions $\mathcal L$ (Lagrangian) or $\mathcal H$ (Hamiltonian). In classical mechanics the Hamiltonian represents the total energy of the system consisting of potential energy $V$ and kinetic energy $T$: $$\mathcal H = T + V\, ,$$ where the functional form of $T$ depends on the setting. In the non-relativistic case it is $$T = \frac{p^2}{2m}\, .$$ The system then evolves such that the potential energy is minimized following the following equations of motion:$$\dot q = \frac{\partial \mathcal H}{\partial p}\, , \quad \textsf{and}\quad \dot p = - \frac{\partial \mathcal H}{\partial q}\, .$$
This already hints the usage as an optimization procedure: Assume a loss function of interest to be the potential energy and the parameters of the model to be the "position variables" $q$. Then initialize a "momentum variable" $p$ for each $q$ and propagate the system according to the equations of motion. The initialization of the momenta is somewhat arbitrary - we chose a gaussian initialization.


### Optimization basics

Optimization methods are characterized by several requirements they place on the function that needs to be optimized (*objective function*), some of which are of relevance for the understanding of HD. For more concise definitions, we refer to [Bierlaire (2015)](http://optimizationprinciplesalgorithms.com/) for a well-written and (comparably) easy to understand introductory textbook on optimization that is freely available.

* **Convexity of the Objective Function**: Convex functions are a family of functions that have desirable properties for optimization methods, mainly the fact that any local optimum we find is certain to be a global optimum as well. In brief, convex functions describe the family of functions for which the connecting line between two points is above the function values at every point. All optimization methods in this notebook are designed to work on convex functions.
* **Constrainedness of the Objective Function**: An optimization is constrained if the objective function is restricted by equality requirements (e.g. $\min f(x) \text{ s.t. } c(x) = 0$) or by inequality requirements (e.g. $\min f(x) \text{ s.t. } c(x) \leq 0$). All optimization methods presented in this notebook perform unconstrained optimization.
* **Differentiability of the Objective Function**: Optimization methods require different degrees of differentiability of the objective function. All optimization methods presented in this notebook require information about the gradient of the objective function, which means that the objective function must be differentiable once (i.e. $f \in C^1$).

In addition to the differences in requirements on the objective function, optimization algorithms differ in their efficiency in terms of iterations $k$ needed to find an optimal solution $\hat x$ (so-called *convergence rate*). We can differentiate four orders of convergence, ranked from fastest to slowest: 

1. **Quadratic**: $\lVert x^{k+1} - \hat x\rVert \leq C\lVert x^k - \hat x\rVert^2$ for some $C< \infty$
2. **Superlinear**: $\lVert x^{k+1} - \hat x \rVert \leq \alpha_k \lVert x^k - \hat x\rVert$ for $\alpha_k \downarrow 0$
3. **Linear**: $\lVert x^{k+1} - \hat x\rVert\leq \alpha \lVert x^k - \hat x \rVert$ for some $\alpha < 1$
4. **Sublinear**: $\lVert x^k - \hat x\rVert \leq \frac C{k^\beta}$, often $\beta \in \{2,1,\frac12\}$




## Hamiltonian Descent methods
---

This section is a summary of the HD methods proposed by Maddison et al. (2018). HD methods describe a set of first-order unconstrained optimization methods (i.e. $f \in C^1$) for convex functions. By incorporating the Hamiltonian Framework, it is possible to use the kinetic energy $k(p_t)$ (with $p_t$ being the momentum of $x$ at time $t$) and it's respective $\nabla k$ to obtain additional information about the objective function $f$. In order to be able to obtain linear convergence, the kinetic energy must be chosen proportional to the convex conjugate of $f(x): k(p) \propto f^*(p) + f^*(-p)$ (with $f^*$ being the convex conjugate of $f$, for intuition see [Le Priol, 2020](https://remilepriol.github.io/dualityviz/)). This assumption can be relaxed to $k(p)\geq \alpha \max\{f^*(p), f^*(-p)\}, \; 0 < \alpha \leq 1$ while maintaining linear convergence. Furthermore, depending on the nature of $f$, $k$ must be chosen appropriately to ensure linear convergence.

An apparent benefit of the *HD* method is that it achieves linear convergence while using a fixed step size. 

For neural networks, the most commonly used method is SGD.

* SGD: Sublinear convergence $\mathcal{O} \left(\dfrac{1}{k}\right)$ with $k = 1, ..., \infty$ being the iteration




## Usage of HaGraD
---