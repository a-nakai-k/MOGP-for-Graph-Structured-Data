import autograd.numpy as np
from autograd import grad
try:
    import matplotlib.pyplot as plt
except:
    pass
import argparse
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument("--kernel", default="rbf", type=str, help='base kernel: matern12, matern32, matern52, rbf')
parser.add_argument("--k", default=6, type=int, help='degree of k-regular graph: 6, 12, 18, 24')
parser = parser.parse_args()

dataname = 'data/synthetic1_k' + str(parser.k) + '.mat'
data = scipy.io.loadmat(dataname)
xn = data['xn']
yn = data['yn']
ttilde = data['ttilde']
xn1T = data['xn1T']
yn1T = data['yn1T']

M = yn.shape[1] # 32
N = yn.shape[0] # 10
num_test = 100
size = 10

def log_likelihood_base(bs = np.ones(M), kappas = np.ones(M), lengthscale = 10., variance = np.var(yn), noise = 0.05):
    # bs = np.log(1. + np.exp(bs))
    kappas = np.log(1. + np.exp(kappas))
    lengthscale = np.log(1. + np.exp(lengthscale))
    variance = np.log(1. + np.exp(variance))
    noise = np.log(1. + np.exp(noise))
    # spatial
    B = np.dot(bs.reshape(-1,1), bs.reshape(1,-1)) + np.diag(kappas)
    # temporal
    if parser.kernel == 'rbf':
        sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn**2,1) - 2*np.dot(xn,xn.T)
        temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
    elif parser.kernel == 'matern12':
        dist = np.abs(xn[:, None, :] - xn).sum(axis=2)
        temporal = np.exp(-(1./lengthscale) * dist)
    elif parser.kernel == 'matern32':
        dist = np.abs(xn[:, None, :] - xn).sum(axis=2)
        temporal = (1. + (np.sqrt(3.) / lengthscale) * dist) * np.exp(-(np.sqrt(3.) / lengthscale) * dist)
    elif parser.kernel == 'matern52':
        dist = np.abs(xn[:, None, :] - xn).sum(axis=2)
        sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn**2,1) - 2*np.dot(xn,xn.T)
        temporal = (1. + (np.sqrt(5.) / lengthscale) * dist + (5./3./(lengthscale**2)) * sqdist) * np.exp(-(np.sqrt(5.) / lengthscale) * dist)
    K = variance*temporal
    # full covariance
    CgN = np.kron(B, K) + (noise * np.eye(M*N))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(M*N))
    alpha_L = np.linalg.solve(L,ttilde)

    log_likelihood = (- 0.5*np.linalg.slogdet(CgN)[1]) - (0.5 * np.matmul(alpha_L.T, alpha_L))
    return log_likelihood

def calc_mse(mu, y):
    return np.mean(np.square(mu - y))

def log_likelihood_posterior(x, mu, sigma):
    size = len(x)
    x_mu = np.matrix(x - mu)
    L = np.linalg.cholesky(sigma + 1e-10*np.eye(size))
    alpha_L = np.linalg.solve(L, x_mu)
    return -(float(size)/2.)*np.log(2.*np.pi) - (1./2.)*np.linalg.slogdet(sigma)[1] - (1./2.) * np.matmul(alpha_L.T, alpha_L)

if __name__ == '__main__':
    print('data: synthetic1, training data: {}, model: ICM, kernel: {}, k: {}'.format(N, parser.kernel, parser.k), flush=True)
    
    bs = np.ones(M)
    kappas = np.ones(M)
    lengthscale = np.mean(np.sum(np.square(xn), axis = 1))
    variance = np.var(yn)
    noise = 0.1 * lengthscale
    rate = 0.001 # alpha
    rate2 = 0.0001 # noise
    rate3 = 1. # lengthscale
    rate4 = 0.001 # variance

    print('log-likelihood =', log_likelihood_base(bs, kappas, lengthscale, variance, noise), flush=True)
    l_old = log_likelihood_base(bs, kappas, lengthscale, variance, noise)
    dl = grad(log_likelihood_base, [0, 1, 2, 3, 4])
    dld = dl(bs, kappas, lengthscale, variance, noise)
    bs += rate *np.array(dld[0])
    kappas += rate *np.array(dld[1])
    lengthscale += rate3 * np.array(dld[2])
    variance += rate4 * np.array(dld[3])
    noise += rate2 *np.array(dld[4])
    l = log_likelihood_base(bs, kappas, lengthscale, variance, noise)
    print('log-likelihood =', l, 'bs =', bs, 'kappas =', kappas, 'l =', lengthscale, 'v =', variance, 'n =', noise, flush=True)
    while np.abs(l_old - l) > 0.001:
        l_old = l.copy()
        dld = dl(bs, kappas, lengthscale, variance, noise)
        bs += rate*np.array(dld[0])
        kappas += rate *np.array(dld[1])
        lengthscale += rate3 * np.array(dld[2])
        variance += rate4 * np.array(dld[3])
        noise += rate2 *np.array(dld[4])
        l = log_likelihood_base(bs, kappas, lengthscale, variance, noise)
        print('log-likelihood =', l, 'bs =', bs, 'kappas =', kappas, 'l =', lengthscale, 'v =', variance, 'n =', noise)

    print('log-likelihood =', log_likelihood_base(bs, kappas, lengthscale, variance, noise), flush=True)
    print('bs =', bs, 'kappas =', kappas, flush=True)
    print('l =', lengthscale, 'v =', variance, 'n =', noise, flush=True)

    ll = []
    mses = []
    kappas, lengthscale, variance, noise = np.log(1. + np.exp(kappas)), np.log(1.+np.exp(lengthscale)), np.log(1.+np.exp(variance)), np.log(1. + np.exp(noise))
    for i in range(1,num_test+1):
        xn1 = xn1T[((i-1)*size):(i*size),:]
        yn1 = yn1T[((i-1)*size):(i*size),:]

        MM = yn1.shape[0] # number of test signals
        B = np.dot(bs.reshape(-1,1), bs.reshape(1,-1)) + np.diag(kappas)
        if parser.kernel == 'rbf':
            sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn**2,1) - 2*np.dot(xn,xn.T)
            temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
            K = variance * temporal
            sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn1**2,1) - 2*np.dot(xn,xn1.T)
            temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
            k = variance * temporal
            sqdist = np.sum(xn1**2, 1).reshape(-1,1) + np.sum(xn1**2,1) - 2*np.dot(xn1,xn1.T)
            temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
            k_star = variance * temporal
        elif parser.kernel == 'matern12':
            dist = np.abs(xn[:, None, :] - xn).sum(axis=2)
            K = variance * np.exp(-(1./lengthscale) * dist)
            dist = np.abs(xn[:, None, :] - xn1).sum(axis=2)
            k = variance * np.exp(-(1./lengthscale) * dist)
            dist = np.abs(xn1[:, None, :] - xn1).sum(axis=2)
            k_star = variance * np.exp(-(1./lengthscale) * dist)
        elif parser.kernel == 'matern32':
            dist = np.abs(xn[:, None, :] - xn).sum(axis=2)
            K = variance * (1. + (np.sqrt(3.) / lengthscale) * dist) * np.exp(-(np.sqrt(3.) / lengthscale) * dist)
            dist = np.abs(xn[:, None, :] - xn1).sum(axis=2)
            k = variance * (1. + (np.sqrt(3.) / lengthscale) * dist) * np.exp(-(np.sqrt(3.) / lengthscale) * dist)
            dist = np.abs(xn1[:, None, :] - xn1).sum(axis=2)
            k_star = variance * (1. + (np.sqrt(3.) / lengthscale) * dist) * np.exp(-(np.sqrt(3.) / lengthscale) * dist)
        elif parser.kernel == 'matern52':
            dist = np.abs(xn[:, None, :] - xn).sum(axis=2)
            sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn**2,1) - 2*np.dot(xn,xn.T)
            K = variance * (1. + (np.sqrt(5.) / lengthscale) * dist + (5./3./(lengthscale**2)) * sqdist) * np.exp(-(np.sqrt(5.) / lengthscale) * dist)
            dist = np.abs(xn[:, None, :] - xn1).sum(axis=2)
            sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn1**2,1) - 2*np.dot(xn,xn1.T)
            k = variance * (1. + (np.sqrt(5.) / lengthscale) * dist + (5./3./(lengthscale**2)) * sqdist) * np.exp(-(np.sqrt(5.) / lengthscale) * dist)
            dist = np.abs(xn1[:, None, :] - xn1).sum(axis=2)
            sqdist = np.sum(xn1**2, 1).reshape(-1,1) + np.sum(xn1**2,1) - 2*np.dot(xn1,xn1.T)
            k_star = variance * (1. + (np.sqrt(5.) / lengthscale) * dist + (5./3./(lengthscale**2)) * sqdist) * np.exp(-(np.sqrt(5.) / lengthscale) * dist)
        
        CgN = np.kron(B, K) + (noise * np.eye(M*N))
        D = np.kron(B, k)
        F = np.kron(B, k_star) + (noise * np.eye(M*MM))

        # cholesky decomposition
        L = np.linalg.cholesky(CgN + 1e-10*np.eye(M*N))

        # posterior mean and sd
        LT = np.linalg.solve(L, ttilde)
        muN1 = D.T.dot(np.linalg.solve(L.T, LT))
        LT = np.linalg.solve(L, D)
        sigmaN1 = F - D.T.dot(np.linalg.solve(L.T,LT))

        log_likelihood = log_likelihood_posterior(yn1.reshape(-1,1, order = 'F'), muN1, sigmaN1)
        mse = calc_mse(muN1, yn1.reshape(-1,1, order = 'F'))
        ll.append((log_likelihood/size).item())
        mses.append(float(mse))

    print(ll, flush=True)
    print(mses, flush=True)
    print('log-likelihood: mean = {}, se = {}'.format(np.mean(ll), np.std(ll)/np.sqrt(len(ll))), flush=True)
    print('mse: mean = {}, se = {}'.format(np.mean(mses), np.std(mses)/np.sqrt(len(mses))), flush=True)