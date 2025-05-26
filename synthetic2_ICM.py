import autograd.numpy as np
from autograd import grad
try:
    import matplotlib.pyplot as plt
except:
    pass
from pygsp import graphs
import argparse
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument("--kernel", default="rbf", type=str, help='base kernel: matern12, matern32, matern52, rbf')
parser = parser.parse_args()

M = 6
N = 20
size = int(N/2)
num_test = 100
noisevar = 0.01

# graph
G2 = graphs.Ring(N=M)
W = G2.W
W[0,5] = 1
W[5,0] = 1
W[1,4] = 1
W[4,1] = 1
G = graphs.Graph(W=W)

# synthetic data 2
def joint_x(start,num):
  return np.sort(start + 2.5*np.random.rand(num))
def joint_x_2(start,num):
  return np.sort(start + 1.25*np.random.rand(num))

data = scipy.io.loadmat('data/synthetic2.mat')
xn = data['xn']
yn = data['yn']
ttilde = data['ttilde']

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
    BB = np.kron(B,np.ones((N,N)))
    CgN = BB[:N*(M-1)+size, :N*(M-1)+size] * K + (noise * np.eye(N*(M-1)+size))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*(M-1)+size))
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
    print('data: synthetic2, training data: {}, baseline: ICM'.format(N), flush=True)
    
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
        xn1 = joint_x_2(6.25,size).reshape(-1,1)
        yn1 = np.sin(xn1)/xn1 + noisevar*np.random.randn(xn1.shape[0],xn1.shape[1])

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
        
        BB = np.kron(B,np.ones((N,N)))
        CgN = BB[:N*(M-1)+size, :N*(M-1)+size] * K + (noise * np.eye(N*(M-1)+size))
        D = BB[:N*(M-1)+size,N*(M-1)+size:] * k
        F = BB[N*(M-1)+size:,N*(M-1)+size:] * k_star + (noise * np.eye(MM))

        # cholesky decomposition
        L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*(M-1)+size))

        # posterior mean and sd
        LT = np.linalg.solve(L, ttilde)
        muN1 = D.T.dot(np.linalg.solve(L.T, LT))
        LT = np.linalg.solve(L, D)
        sigmaN1 = F - D.T.dot(np.linalg.solve(L.T,LT))

        log_likelihood = log_likelihood_posterior(yn1, muN1, sigmaN1)
        mse = calc_mse(muN1, yn1)
        ll.append((log_likelihood/size).item())
        mses.append(float(mse))

    print(ll, flush=True)
    print(mses, flush=True)
    print('log-likelihood: mean = {}, se = {}'.format(np.mean(ll), np.std(ll)/np.sqrt(len(ll))), flush=True)
    print('mse: mean = {}, se = {}'.format(np.mean(mses), np.std(mses)/np.sqrt(len(mses))), flush=True)