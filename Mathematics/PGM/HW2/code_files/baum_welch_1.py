import numpy as np
from scipy.stats import multivariate_normal


def e_step(X, pi, A, mu, sigma2):
    """E-step: forward-backward message passing"""
    # Messages and sufficient statistics
    N, T, K = X.shape
    M = A.shape[0]
    alpha = np.zeros([N, T, M])  # [N,T,M]
    alpha_sum = np.zeros([N, T])  # [N,T], normalizer for alpha
    beta = np.zeros([N, T, M])  # [N,T,M]
    gamma = np.zeros([N, T, M])  # [N,T,M]
    xi = np.zeros([N, T-1, M, M])  # [N,T-1,M,M]

    # Forward messages
    emission_probabilities = np.stack([multivariate_normal.pdf(X, mean=mu[m], cov=sigma2[m] * np.eye(K)) for m in range(M)], axis=2)
    # Initialize alpha at t=0
    alpha[:, 0, :] = pi * emission_probabilities[:, 0, :]  # [N,M]
    alpha_sum[:, 0] = np.sum(alpha[:, 0, :], axis=1)  # [N,]
    alpha[:, 0, :] = alpha[:, 0, :] / alpha_sum[:, 0, np.newaxis]  # Normalize

    # Forward pass
    for t in range(1, T):
        for m in range(M):
            alpha[:, t, m] = emission_probabilities[:, t, m] * \
                np.sum(alpha[:, t-1, :] * A[:, m], axis=1)

        alpha_sum[:, t] = np.sum(alpha[:, t, :], axis=1)  # [N,]
        alpha[:, t, :] = alpha[:, t, :] / \
            alpha_sum[:, t, np.newaxis]  # Normalize

    # Backward messages
    # Initialize beta at t=T-1
    beta[:, T-1, :] = 1.0

    # Backward pass
    for t in range(T-2, -1, -1):
        for m in range(M):
            for n in range(N):
                beta[n, t, m] = np.sum(
                    A[m, :] * emission_probabilities[n, t+1, :] * beta[n, t+1, :])

        
        beta[:, t, :] = beta[:, t, :] / \
            np.sum(beta[:, t, :], axis=1, keepdims=True)

    # Compute gamma (posterior state probabilities)
    for t in range(T):
        gamma[:, t, :] = alpha[:, t, :] * beta[:, t, :]
        # Normalize gamma
        gamma[:, t, :] = gamma[:, t, :] / \
            np.sum(gamma[:, t, :], axis=1, keepdims=True)

    # Compute xi (posterior transition probabilities)
    for t in range(T-1):
        for n in range(N):
            for i in range(M):
                for j in range(M):
                    xi[n, t, i, j] = alpha[n, t, i] * A[i, j] * emission_probabilities[n, t+1, j] * beta[n, t+1, j]

            # Normalize xi for each sequence and time step
            xi[n, t] = xi[n, t] / np.sum(xi[n, t])

    # Although some of them will not be used in the M-step, please still
    # return everything as they will be used in test cases
    return alpha, alpha_sum, beta, gamma, xi


def m_step(X, gamma, xi):
    """M-step: MLE"""
    N, T, K = X.shape
    M = gamma.shape[2]

    # Updating initial state distribution pi
    pi = np.sum(gamma[:, 0, :], axis=0) / N

    # Updating transition matrix A
    A = np.sum(xi, axis=(0, 1))  # Sum over N and T
    A /= np.sum(A, axis=1, keepdims=True)  # Normalize rows

    # Updating emission parameters mu and sigma2
    state = np.sum(gamma, axis=(0, 1))  # Shape: (M,)

    # Update mean vectors mu
    mu = np.einsum('ntm,ntk->mk', gamma, X) / state[:, np.newaxis]

    sigma2 = np.einsum('ntm,ntmk->m', gamma, (X[:, :, np.newaxis, :] - mu[np.newaxis, np.newaxis, :, :]) ** 2) / (state * K)
    return pi, A, mu, sigma2

def hmm_train(X, pi, A, mu, sigma2, em_step=20):
    """Run Baum-Welch algorithm."""
    for step in range(em_step):
        alpha, alpha_sum, beta, gamma, xi = e_step(X, pi, A, mu, sigma2)
        pi, A, mu, sigma2 = m_step(X, gamma, xi)
        print(f"step: {step}  ln p(x): {np.einsum('nt->', np.log(alpha_sum))}")
    return pi, A, mu, sigma2


def hmm_generate_samples(N, T, pi, A, mu, sigma2):
    """Given pi, A, mu, sigma2, generate [N,T,K] samples."""
    M, K = mu.shape
    Y = np.zeros([N, T], dtype=int)
    X = np.zeros([N, T, K], dtype=float)
    for n in range(N):
        Y[n, 0] = np.random.choice(M, p=pi)  # [1,]
        X[n, 0, :] = multivariate_normal.rvs(
            mu[Y[n, 0], :], sigma2[Y[n, 0]] * np.eye(K))  # [K,]
    for t in range(T - 1):
        for n in range(N):
            Y[n, t+1] = np.random.choice(M, p=A[Y[n, t], :])  # [1,]
            # [K,]
            X[n, t+1, :] = multivariate_normal.rvs(
                mu[Y[n, t+1], :], sigma2[Y[n, t+1]] * np.eye(K))
    return X


def main():
    """Run Baum-Welch on a simulated toy problem."""
    # Generate a toy problem
    np.random.seed(12345)  # for reproducibility
    N, T, M, K = 10, 100, 4, 2
    pi = np.array([.0, .0, .0, 1.])  # [M,]
    A = np.array([[.7, .1, .1, .1],
                  [.1, .7, .1, .1],
                  [.1, .1, .7, .1],
                  [.1, .1, .1, .7]])  # [M,M]
    mu = np.array([[2., 2.],
                   [-2., 2.],
                   [-2., -2.],
                   [2., -2.]])  # [M,K]
    sigma2 = np.array([.2, .4, .6, .8])  # [M,]
    X = hmm_generate_samples(N, T, pi, A, mu, sigma2)

    # Run on the toy problem
    pi_init = np.random.rand(M)
    pi_init = pi_init / pi_init.sum()
    A_init = np.random.rand(M, M)
    A_init = A_init / A_init.sum(axis=-1, keepdims=True)
    mu_init = 2 * np.random.rand(M, K) - 1
    sigma2_init = np.ones(M)

    pi, A, mu, sigma2 = hmm_train(
        X, pi_init, A_init, mu_init, sigma2_init, em_step=20)
    print(pi)
    print(A)
    print(mu)
    print(sigma2)


if __name__ == '__main__':
    main()