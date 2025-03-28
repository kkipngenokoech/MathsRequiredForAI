"""Dataset utilities.

Do not modify the contents of this file.
"""

import numpy as np


class Dataset:
    """Dataset iterator.

    Parameters
    ----------
    X : np.array
        Input data points; should have shape (dataset size, features).
    y : labels
        Output one-hot labels; should have shape (dataset size, classes).
    """

    def __init__(self, X, y, batch_size=16):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.size = X.shape[0] // batch_size

    def __iter__(self):
        self.idx = 0
        self.indices = np.random.permutation(
            self.X.shape[0]
        )[:self.size * self.batch_size].reshape(self.size, self.batch_size)
        return self

    def __next__(self):
        if self.idx < self.size:
            batch = self.indices[self.idx]
            self.idx += 1
            return (self.X[batch], self.y[batch])
        else:
            raise StopIteration()


def load_mnist(filepath):
    """Load mnist-style dataset.

    Parameters
    ----------
    filepath : str
        Target filepath. Should be a .npz file.

    Returns
    -------
    (np.array, np.array)
        [0] Loaded images scaled to 0-1.
        [1] Loaded labels one-hot encoded.
    """
    data = np.load(filepath)
    X = data['image'].astype(np.float32) / 255
    y = np.eye(10)[data['label']]

    return X, y
