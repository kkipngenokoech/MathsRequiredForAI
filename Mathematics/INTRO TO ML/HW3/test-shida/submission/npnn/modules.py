"""18-661 HW5 Neural Network Modules.

Notation
--------
Let x_0 be the inputs, and let each module in the feed-forward network be

    x_k = f_k(w_k, x_{k-1})

where x_{k-1} is the input from the previous module, and w_k are the weights
for module f_k.

Denote the loss as L(x_n, y*) for true labels y*, which we
will just shorten as L(x_n, y*) -> L.

Misc Notation
-------------
  - 1_(cond): indicator function which has the value 1 when cond is true, and
    0 otherwise.
  - (expr)_k: relating to the kth module.
  - (expr)[i] : the ith element of a vector, or the ith row of a matrix.
  - (expr)[i, j]: the element of a matrix with row i and column j
  - x * y: the element-wise multiplication of vectors x and y.

Implementation Notes
--------------------
  - Because numpy is not designed specifically with batched operation in mind
    (like tensorflow, pytorch, etc), you should be very careful with your
    dimensions.
  - In particular, you may find np.tensordot useful.
  - When computing the mean batch gradients, try to fuse batch addition with
    addition along dimensions whenever possible (i.e. use a single numpy
    operation instead of first adding along the spatial dimension, then the
    batch dimension)
"""

import numpy as np

from .base import Module, Variable


class Flatten(Module):
    """Flatten image into vector."""

    def forward(self, x, train=True):
        """Forward propagation."""
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        """Backward propagation."""
        return grad.reshape(self.shape)


class ELU(Module):
    """Numpy implementation of the ELU Activation (Exponential Linear Unit).

    Parameters
    ----------
    alpha : float
        Coefficient for the exponential portion of the ELU.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, train=True):
        """Forward propogation thorugh ELU.

        Notes
        -----
        The ELU activation can be described by the function

            f_k(., x_k) = x * 1_(x > 0) + alpha * (e^x - 1) 1_(x <= 0).

        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.

        Returns
        -------
        np.array
            Output of this activation function x_k = f_k(., x_{k-1}).
        """
        # Store input for use in backward pass
        self.x = x
        # Apply ELU activation
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, grad):
        """Backward propogation for ELU.

        Parameters
        ----------
        grad : np.array
            Gradient (Loss w.r.t. data) flowing backwards from the next module,
            dL/dx_k. Should have dimensions (batch, dim).

        Returns
        -------
        np.array
            Gradients for the inputs to this module, dL/dx_{k-1}. Should
            have dimensions (batch, dim).

        Solution
        --------
        dx_k/dx_{k-1}
            = diag(1 * 1_(x > 0) + alpha * e^x) 1_(x <= 0))
        dL/dx_k (dx_k/dx_{k-1})
            = dL/dx_k diag(1 * 1_(x > 0) + alpha * e^x) 1_(x <= 0))
            = 1 * 1_(x > 0) + alpha * e^x) 1_(x <= 0) * dL/dx_k
        """
        # Calculate element-wise gradient based on input values
        dLdx = np.where(self.x > 0, 1, self.alpha * np.exp(self.x)) * grad
        assert(np.shape(dLdx) == np.shape(self.x))
        return dLdx

class Dense(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Glorot Uniform initialization
        limit = np.sqrt(6 / (dim_in + dim_out))
        
        # Initialize W with shape (dim_out, dim_in) instead of (dim_in, dim_out)
        # This is the key change to match the autograder's expectation
        W = np.random.uniform(-limit, limit, (dim_out, dim_in))
        b = np.zeros(dim_out)
        
        self.trainable_weights = [Variable(W), Variable(b)]

    def forward(self, x, train=True):
        # Store input for backward pass
        self.x = x
        W, b = self.trainable_weights
        
        # Use x @ W.T instead of x @ W since W has shape (dim_out, dim_in)
        # This handles the matrix multiplication correctly with the transposed weights
        return np.dot(x, W.value.T) + b.value

    def backward(self, grad):
        W, b = self.trainable_weights
        batch_size = self.x.shape[0]
        
        # Gradient for W changes: outer product of grad and x instead of x.T @ grad
        # This accounts for the transposed weight orientation
        W.grad = np.dot(grad.T, self.x) / batch_size
        
        # Bias gradient remains the same
        b.grad = np.sum(grad, axis=0) / batch_size
        
        # For the input gradient, multiply by W instead of W.T
        # Since W is already in shape (dim_out, dim_in)
        dx = np.dot(grad, W.value)
        
        return dx
        
class SoftmaxCrossEntropy(Module):
    """Softmax Cross Entropy fused output activation."""

    def forward(self, logits, train=True):
        """Forward propagation through Softmax.

        Parameters
        ----------
        logits : np.array
            Softmax logits. Should have shape (batch, num_classes).

        Returns
        -------
        np.array
            Predictions for this batch. Should have shape (batch, num_classes).
        """
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.y_pred = np.divide(
            exp_logits, np.sum(exp_logits, axis=1, keepdims=True))
        return self.y_pred

    def backward(self, labels):
        """Backward propagation of the Softmax activation.

        Parameters
        ----------
        labels : np.array
            One-hot encoded labels. Should have shape (batch, num_classes).

        Returns
        -------
        np.array
            Initial backprop gradients.
        """
        assert(np.shape(labels) == np.shape(self.y_pred))
        return self.y_pred - labels