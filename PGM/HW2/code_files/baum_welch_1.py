import numpy as np
from scipy.stats import multivariate_normal


def e_step(X, pi, A, mu, sigma2):
    """E-step: forward-backward message passing"""
    # Messages and sufficient statistics
    N, T, K = X.shape
    M = A.shape[0]
    alpha = np.zeros([N,T,M])  # [N,T,M]
    alpha_sum = np.zeros([N,T])  # [N,T], normalizer for alpha
    beta = np.zeros([N,T,M])  # [N,T,M]
    gamma = np.zeros([N,T,M])  # [N,T,M]
    xi = np.zeros([N,T-1,M,M])  # [N,T-1,M,M]

    # Forward messages
    # TODO ...
    for n in range(N):
        for t in range(T):
            if t == 0:
                alpha[n,t,:] = pi * multivariate_normal.pdf(X[n,t,:], mu, sigma2)
            else:
                alpha[n,t,:] = np.einsum('m,ml->l', alpha[n,t-1,:], A) * multivariate_normal.pdf(X[n,t,:], mu, sigma2)
            alpha_sum[n,t] = alpha[n,t,:].sum()
            alpha[n,t,:] = alpha[n,t,:] / alpha_sum[n,t]
            
    

    # Backward messages
    # TODO ...
    for n in range(N):
        for t in range(T-1, -1, -1):
            if t == T-1:
                beta[n,t,:] = 1
            else:
                beta[n,t,:] = np.einsum('m,ml,l->m', beta[n,t+1,:], A, multivariate_normal.pdf(X[n,t+1,:], mu, sigma2))
            beta[n,t,:] = beta[n,t,:] / alpha_sum[n,t]
            

    # Sufficient statistics
    # TODO ...
    for n in range(N):
        for t in range(T):
            gamma[n,t,:] = alpha[n,t,:] * beta[n,t,:]
            gamma[n,t,:] = gamma[n,t,:] / gamma[n,t,:].sum()
        for t in range(T-1):
            xi[n,t,:,:] = np.outer(alpha[n,t,:], beta[n,t+1,:]) * A * multivariate_normal.pdf(X[n,t+1,:], mu, sigma2)
            xi[n,t,:,:] = xi[n,t,:,:] / xi[n,t,:,:].sum()
    # Although some of them will not be used in the M-step, please still
    # return everything as they will be used in test cases
    return alpha, alpha_sum, beta, gamma, xi


import numpy as np

def m_step(X, gamma, xi):
    """M-step: MLE"""
    
    N, T, M = gamma.shape
    _, _, M, _ = xi.shape
    
    # 1. Update the initial state distribution (pi)
    pi = gamma[:, 0, :].sum(axis=0) / N
    
    # 2. Update the transition matrix (A)
    A = xi.sum(axis=0) / xi.sum(axis=(0, 1)).reshape(M, 1)  # Normalize
    
    # 3. Update the mean vectors (mu)
    mu = np.einsum('ntk,ntm->mk', X, gamma) / gamma.sum(axis=0).sum(axis=0).reshape(M, 1)
    
    # 4. Update the covariance (sigma2)
    diff = X - mu.reshape(1, M, -1)
    sigma2 = np.einsum('ntk,ntm->m', (diff ** 2), gamma) / gamma.sum(axis=0).sum(axis=0)
    
    return pi, A, mu, sigma2



def hmm_train(X, pi, A, mu, sigma2, em_step=20):
    """Run Baum-Welch algorithm."""
    for step in range(em_step):
        _, alpha_sum, _, gamma, xi = e_step(X, pi, A, mu, sigma2)
        pi, A, mu, sigma2 = m_step(X, gamma, xi)
        print(f"step: {step}  ln p(x): {np.einsum('nt->', np.log(alpha_sum))}")
    return pi, A, mu, sigma2


def hmm_generate_samples(N, T, pi, A, mu, sigma2):
    """Given pi, A, mu, sigma2, generate [N,T,K] samples."""
    M, K = mu.shape
    Y = np.zeros([N,T], dtype=int) 
    X = np.zeros([N,T,K], dtype=float)
    for n in range(N):
        Y[n,0] = np.random.choice(M, p=pi)  # [1,]
        X[n,0,:] = multivariate_normal.rvs(mu[Y[n,0],:], sigma2[Y[n,0]] * np.eye(K))  # [K,]
    for t in range(T - 1):
        for n in range(N):
            Y[n,t+1] = np.random.choice(M, p=A[Y[n,t],:])  # [1,]
            X[n,t+1,:] = multivariate_normal.rvs(mu[Y[n,t+1],:], sigma2[Y[n,t+1]] * np.eye(K))  # [K,]
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
    pi, A, mu, sigma2 = hmm_train(X, pi_init, A_init, mu_init, sigma2_init, em_step=20)
    print(pi)
    print(A)
    print(mu)
    print(sigma2)


if __name__ == '__main__':
    main()

