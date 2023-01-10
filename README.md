# HaGraD

This repository presents an implementation of one of the Hamiltonial Descent Methods presented in 

Maddison, C. J., Paulin, D., Teh, Y. W., O'Donoghue, B., & Doucet, A. (2018). *Hamiltonian Descent Methods*. arXiv:[1809.05042](https://arxiv.org/abs/1809.05042)

We implement the first explicit method (p. 17) as a [`keras.optimizers.Optimizer`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer) for usage with keras models. We were guided by [cloudxlab.com/blog/writing-custom-optimizer-in-tensorflow-and-keras/](https://cloudxlab.com/blog/writing-custom-optimizer-in-tensorflow-and-keras/).