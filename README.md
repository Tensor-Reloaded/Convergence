# Convergence
Artificial Neural Network training convergence boosting by smart data ordering.

The project relies on the assumtion that there must be an optimal order in which training data is fed to a model.
We assume this order would improve the convergence speed of the training process, effectively requiring less time and computing power.

This optimal order will be approximated using metrics collected dynamicly during training, such as sample/batch loss, gradients, input-ouput distance etc.

This convergence boosting method is promissing since it could be applied to any ANN during training in order to speed it up  
