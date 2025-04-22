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
    def __init__(self, modules, loss=None, optimizer=None):
        # Check types
        for module in modules:
            assert isinstance(module, Module)
        
        if loss is not None:
            assert isinstance(loss, Module)
        if optimizer is not None:
            assert isinstance(optimizer, Optimizer)
            
        self.modules = modules
        self.loss = loss
        
        # Collect trainable parameters
        self.params = []
        for module in modules:
            self.params += module.trainable_weights
            
        self.optimizer = optimizer
        if optimizer is not None:
            self.optimizer.initialize(self.params)

    def forward(self, X, train=True):
        # Forward pass through each module
        for module in self.modules:
            X = module.forward(X, train)
            
        # Forward pass through loss if provided
        if self.loss is not None:
            X = self.loss.forward(X, train)
            
        return X

    def backward(self, y):
        # Backward pass through loss
        if self.loss is not None:
            grad = self.loss.backward(y)
        else:
            grad = y
            
        # Backward pass through modules in reverse order
        for module in reversed(self.modules):
            grad = module.backward(grad)
            
        # Apply gradients if optimizer is provided
        if self.optimizer is not None:
            self.optimizer.apply_gradients(self.params)
            
        return grad

    def train(self, dataset):
        total_loss = 0
        total_acc = 0
        count = 0
        
        for X, y in dataset:
            # Forward pass
            pred = self.forward(X)
            # Calculate loss and accuracy
            loss = categorical_cross_entropy(pred, y)
            acc = categorical_accuracy(pred, y)
            # Backward pass
            self.backward(y)
            
            total_loss += loss
            total_acc += acc
            count += 1
            
        return total_loss / count, total_acc / count

    def test(self, dataset):
        total_loss = 0
        total_acc = 0
        count = 0
        
        for X, y in dataset:
            # Forward pass (with train=False)
            pred = self.forward(X, train=False)
            # Calculate loss and accuracy
            loss = categorical_cross_entropy(pred, y)
            acc = categorical_accuracy(pred, y)
            
            total_loss += loss
            total_acc += acc
            count += 1
            
        return total_loss / count, total_acc / count
    