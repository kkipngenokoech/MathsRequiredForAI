import ipdb

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
    """Numpy implementation of Dense Layer.

    Parameters
    ----------
    dim_in : int
        Number of input dimensions.
    dim_out : int
        Number of output dimensions.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Glorot Uniform initialization
        limit = np.sqrt(6 / (dim_in + dim_out))
        W = np.random.uniform(-limit, limit, (dim_in, dim_out))
        b = np.zeros(dim_out)

        self.trainable_weights = [Variable(W), Variable(b)]

    def forward(self, x, train=True):
        """Forward propagation for a Dense layer.

        In vectorized form, the output is given as

            x_k = f_k((W_k, b_k), x_{k-1}) = W_kx_{k-1} + b_k.

        You may find it helpful to also think about the dense layer in
        per-feature terms, namely

            x_k[a] = sum_b W_k[a, b] x_{k-1}[b].

        Parameters
        ----------
        x : np.array
            Input for this layer x. Should have dimensions (batch, dim).

        Returns
        -------
        np.array
            Output of this layer f(w, x) for weights w. Should have dimensions
            (batch, dim).
        """
        # Store input for backward pass
        self.x = x
        W, b = self.trainable_weights
        # Calculate output: X·W + b
        return np.dot(x, W.value) + b.value

    def backward(self, grad):
        """Backward propagation for a Dense layer.

        Should set ```self.trainable_weights[*].grad``` to the mean batch
        gradients (1) for the trainable weights in this layer,

            E[dL/dw_k] = E[dL/dx_k dx_k/dw_k] (2),

        and return the gradients flowing to the previous layer,

            dL/dx_{k-1} = dL/dx_k (dx_k/dx_{k-1}).

        Parameters
        ----------
        grad : np.array
            Gradient (Loss w.r.t. data) flowing backwards from the next layer,
            dL/dx_k. Should have dimensions (batch, dim).

        Returns
        -------
        np.array
            Gradients for the inputs to this layer, dL/dx_{k-1}. Should
            have dimensions (batch, dim).
        """
        W, b = self.trainable_weights
        batch_size = self.x.shape[0]
        
        # Calculate gradients for weights: (X^T · grad) / batch_size
        W.grad = np.dot(self.x.T, grad) / batch_size
        
        # Calculate gradients for bias: sum(grad, axis=0) / batch_size
        b.grad = np.sum(grad, axis=0) / batch_size
        
        # Calculate gradients for inputs: grad · W^T
        dx = np.dot(grad, W.value.T)
        
        # Verify shapes
        assert(np.shape(self.x) == np.shape(dx))
        assert(np.shape(W.value) == np.shape(W.grad))
        assert(np.shape(b.value) == np.shape(b.grad))
        
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