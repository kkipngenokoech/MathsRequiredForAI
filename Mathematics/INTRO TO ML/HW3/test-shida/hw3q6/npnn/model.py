"""Neural Network model."""

from .modules import Module
from .optimizer import Optimizer

import numpy as np


def categorical_cross_entropy(pred, labels, epsilon=1e-10):
    """Cross entropy loss function.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).
    epsilon : float
        Small constant to add to the log term of cross entropy to help
        with numerical stability.

    Returns
    -------
    float
        Cross entropy loss.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))


def categorical_accuracy(pred, labels):
    """Accuracy statistic.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).

    Returns
    -------
    float
        Mean accuracy in this batch.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))


class Sequential:
    """Sequential neural network model.

    Parameters
    ----------
    modules : Module[]
        List of modules; used to grab trainable weights.
    loss : Module
        Final output activation and loss function.
    optimizer : Optimizer
        Optimization policy to use during training.
    """

    def __init__(self, modules, loss=None, optimizer=None):
        for module in modules:
            assert(isinstance(module, Module))
            
        if loss is not None:
            assert(isinstance(loss, Module))
        if optimizer is not None:
            assert(isinstance(optimizer, Optimizer))

        self.modules = modules
        self.loss = loss

        self.params = []
        for module in modules:
            self.params += module.trainable_weights

        self.optimizer = optimizer
        if optimizer is not None:
            self.optimizer.initialize(self.params)

    def forward(self, X, train=True):
        """Model forward pass.

        Parameters
        ----------
        X : np.array
            Input data

        Keyword Args
        ------------
        train : bool
            Indicates whether we are training or testing.

        Returns
        -------
        np.array
            Batch predictions; should have shape (batch, num_classes).
        """
        # Forward pass through each module
        for module in self.modules:
            X = module.forward(X, train)
            
        # Forward pass through loss function if provided
        if self.loss is not None:
            X = self.loss.forward(X, train)
            
        return X

    def backward(self, y):
        """Model backwards pass.

        Parameters
        ----------
        y : np.array
            True labels.
        """
        # Backward pass through loss function
        if self.loss is not None:
            grad = self.loss.backward(y)
        else:
            grad = y
            
        # Backward pass through modules in reverse order
        for module in reversed(self.modules):
            grad = module.backward(grad)
            
        # Apply gradients to parameters using optimizer
        if self.optimizer is not None:
            self.optimizer.apply_gradients(self.params)
            
        return grad

    def train(self, dataset):
        """Fit model on dataset for a single epoch.

        Parameters
        ----------
        dataset : Dataset
            Training dataset with batches already split.

        Returns
        -------
        (float, float)
            [0] Mean train loss during this epoch.
            [1] Mean train accuracy during this epoch.
        """
        loss = 0
        accuracy = 0
        num_batches = 0
        
        for X, y in dataset:
            # Forward pass
            pred = self.forward(X)
            # Calculate loss and accuracy
            loss += categorical_cross_entropy(pred, y)
            accuracy += categorical_accuracy(pred, y)
            # Backward pass
            self.backward(y)
            num_batches += 1
            
        # Return average loss and accuracy
        return loss / num_batches, accuracy / num_batches

    def test(self, dataset):
        """Compute test/validation loss for dataset.

        Parameters
        ----------
        dataset : Dataset
            Validation dataset with batches already split.

        Returns
        -------
        (float, float)
            [0] Mean test loss.
            [1] Test accuracy.
        """
        loss = 0
        accuracy = 0
        num_batches = 0
        
        for X, y in dataset:
            # Forward pass (with train=False)
            pred = self.forward(X, train=False)
            # Calculate loss and accuracy
            loss += categorical_cross_entropy(pred, y)
            accuracy += categorical_accuracy(pred, y)
            num_batches += 1
            
        # Return average loss and accuracy
        return loss / num_batches, accuracy / num_batches