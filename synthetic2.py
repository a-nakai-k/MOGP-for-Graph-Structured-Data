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
parser.add_argument("--model", default="standard", type=str, help=' graph kernel: standard, laplacian, local_averaging, global_filtering, regularized_laplacian, diffusion, 1_random_walk, 3_random_walk, cosine, matern2, matern3, matern5, gf_3random, gf_1random, gf_diff, reg_matern3, loc_matern2')
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

if parser.model in ('standard', 'regularized_laplacian', 'diffusion', '1_random_walk', '3_random_walk', 'cosine', 'matern2', 'matern3', 'matern5'):
    G.compute_laplacian('normalized')
    G.compute_fourier_basis('normalized')
elif parser.model in ('gf_3random', 'loc_matern2', 'gf_1random', 'gf_diff'):
    G.compute_laplacian()
    G.compute_fourier_basis()
    L1 = G.L.todense()
    W1 = G.W.todense()
    d1 = G.d
    G.compute_laplacian('normalized')
    G.compute_fourier_basis('normalized')
    L2 = G.L.todense()
    U2 = G.U
    e2 = G.e
else:
    G.compute_laplacian()
    G.compute_fourier_basis()

def log_likelihood_base(alpha_log = 1., beta_log = 1., lengthscale = 10., variance = np.var(yn), noise = 0.05):
    alpha = np.log(1. + np.exp(alpha_log))
    beta = np.log(1. + np.exp(beta_log))
    lengthscale = np.log(1. + np.exp(lengthscale))
    variance = np.log(1. + np.exp(variance))
    noise = np.log(1. + np.exp(noise))
    # spatial
    I = np.eye(M)
    global B
    global L1, W1, d1, L2, U2, e2
    if parser.model == 'standard':
        B = I
    elif parser.model == 'laplacian':
        B = np.matmul(np.matmul(G.U, np.diag(np.concatenate((np.array([0]), np.sqrt(1./G.e[1:]))))), G.U.T)
    elif parser.model == 'local_averaging':
        B = np.matmul(np.linalg.inv(I + alpha*np.diag(G.d)), I + alpha*np.array(G.W.todense()))
    elif parser.model == 'global_filtering':
        B = np.linalg.inv((I + alpha*np.array(G.L.todense())))
    elif parser.model == 'regularized_laplacian':
        B = np.linalg.inv(I + alpha*np.array(G.L.todense()))
    elif parser.model == 'diffusion':
        B = np.matmul(G.U, np.matmul(np.diag(np.exp(-alpha*G.e)), G.U.T))
    elif parser.model == '1_random_walk': 
        B = (2+alpha)*I + np.array(G.L.todense())
    elif parser.model == '3_random_walk':
        B = np.matmul(G.U, np.matmul(np.diag(((2 + alpha) - G.e)**3), G.U.T))
    elif parser.model == 'cosine':
        B = np.matmul(G.U, np.matmul(np.diag(np.cos(np.pi*G.e/4.)), G.U.T))
    elif parser.model == 'gf_3random':
        B1 = np.linalg.solve(I + alpha*np.array(L1), I)
        B2 = np.matmul(U2, np.matmul(np.diag(((2 + beta) - e2)**3), U2.T))
    elif parser.model == 'gf_1random':
        B1 = np.linalg.solve(I + alpha*np.array(L1), I)
        B2 = (2+beta)*I + np.array(L2)
    elif parser.model == 'gf_diff':
        B1 = np.linalg.solve(I + alpha*np.array(L1), I)
        B2 = np.matmul(U2, np.matmul(np.diag(np.exp(-beta*e2)), U2.T))
    elif parser.model == 'reg_diff':
        B1 = np.linalg.inv(I + alpha*np.array(G.L.todense()))
        B2 = np.matmul(G.U, np.matmul(np.diag(np.exp(-beta*G.e)), G.U.T))
        B = B1 + B2
    elif parser.model == 'reg_matern3':
        B1 = np.linalg.inv(I + alpha*np.array(G.L.todense()))
        inv_mat = np.linalg.inv(2*3/beta*I + np.array(G.L.todense()))
        B2 = np.matmul(np.matmul(inv_mat, inv_mat), inv_mat)
        B = B1 + B2
    elif parser.model == 'loc_matern2':
        B1 = np.matmul(np.linalg.inv(I + alpha*np.diag(d1)), I + alpha*np.array(W1))
        inv_mat = np.linalg.inv(2*2/beta*I + np.array(L2))
        B2 = np.matmul(np.matmul(inv_mat, inv_mat), inv_mat)
    elif parser.model == 'matern2':
        inv_mat = np.linalg.inv(2*2/alpha*I + np.array(G.L.todense()))
        B = np.matmul(inv_mat, inv_mat)
    elif parser.model == 'matern3':
        inv_mat = np.linalg.inv(2*3/alpha*I + np.array(G.L.todense()))
        B = np.matmul(np.matmul(inv_mat, inv_mat), inv_mat)
    elif parser.model == 'matern5':
        inv_mat = np.linalg.inv(2*5/alpha*I + np.array(G.L.todense()))
        B = np.matmul(np.matmul(np.matmul(inv_mat, inv_mat), inv_mat), np.matmul(inv_mat, inv_mat))
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
    if parser.model in ('regularized_laplacian', 'diffusion', '1_random_walk', '3_random_walk', 'cosine', 'reg_diff', 'matern2', 'matern3', 'matern5', 'reg_matern3'):
        BB = np.kron(B,np.ones((N,N)))
        CgN = BB[:N*(M-1)+size, :N*(M-1)+size] * K + (noise * np.eye(N*(M-1)+size))
    elif parser.model in ('gf_3random', 'loc_matern2', 'gf_1random', 'gf_diff'):
        BB1 = np.kron(np.matmul(B1,B1.T),np.ones((N,N)))
        BB2 = np.kron(B2,np.ones((N,N)))
        BB = BB1 + BB2
        CgN = BB[:N*(M-1)+size, :N*(M-1)+size] * K + (noise * np.eye(N*(M-1)+size))
    else:
        BB = np.kron(np.matmul(B,B.T),np.ones((N,N)))
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
    print('data: synthetic2, training data: {}, model: {}, kernel: {}'.format(N, parser.model, parser.kernel), flush=True)
    
    alpha = np.log(1.)
    beta = np.log(1.)
    lengthscale = np.mean(np.sum(np.square(xn), axis = 1))
    variance = np.var(yn)
    noise = 0.1 * lengthscale
    rate = 0.001 # alpha
    rate2 = 0.0001 # noise
    rate3 = 1. # lengthscale
    rate4 = 0.001 # variance

    print('log-likelihood =', log_likelihood_base(alpha, beta, lengthscale, variance, noise), flush=True)
    l_old = log_likelihood_base(alpha, beta, lengthscale, variance, noise)
    if parser.model in ('reg_diff', 'gf_3random', 'reg_matern3', 'loc_matern2', 'gf_1random', 'gf_diff'):
        dl = grad(log_likelihood_base, [0, 1, 2, 3, 4])
        dld = dl(alpha, beta, lengthscale, variance, noise)
        alpha += rate *np.array(dld[0])
        beta += rate *np.array(dld[1])
        lengthscale += rate3 * np.array(dld[2])
        variance += rate4 * np.array(dld[3])
        noise += rate2 *np.array(dld[4])
    else:
        dl = grad(log_likelihood_base, [0, 2, 3, 4])
        dld = dl(alpha, beta, lengthscale, variance, noise)
        # print(dld)
        alpha += rate *np.array(dld[0])
        lengthscale += rate3 * np.array(dld[1])
        variance += rate4 * np.array(dld[2])
        noise += rate2 *np.array(dld[3])
    l = log_likelihood_base(alpha, beta, lengthscale, variance, noise)
    if parser.model in ('reg_diff', 'gf_3random', 'reg_matern3', 'loc_matern2', 'gf_1random', 'gf_diff'):
        print('log-likelihood =', l, 'alpha =', alpha, 'beta =', beta, 'l =', lengthscale, 'v =', variance, 'n =', noise, flush=True)
    else:
        print('log-likelihood =', l, 'alpha =', alpha, 'l =', lengthscale, 'v =', variance, 'n =', noise, flush=True)
    while np.abs(l_old - l) > 0.001:
        l_old = l.copy()
        if parser.model in ('reg_diff', 'gf_3random', 'reg_matern3', 'loc_matern2', 'gf_1random', 'gf_diff'):
            dld = dl(alpha, beta, lengthscale, variance, noise)
            alpha += rate*np.array(dld[0])
            beta += rate *np.array(dld[1])
            lengthscale += rate3 * np.array(dld[2])
            variance += rate4 * np.array(dld[3])
            noise += rate2 *np.array(dld[4])
        else:
            dld = dl(alpha, beta, lengthscale, variance, noise)
            alpha += rate*np.array(dld[0])
            lengthscale += rate3 * np.array(dld[1])
            variance += rate4 * np.array(dld[2])
            noise += rate2 *np.array(dld[3])
        l = log_likelihood_base(alpha, beta, lengthscale, variance, noise)
        if parser.model in ('reg_diff', 'gf_3random', 'reg_matern3', 'loc_matern2', 'gf_1random', 'gf_diff'):
            print('log-likelihood =', l, 'alpha =', alpha, 'beta =', beta, 'l =', lengthscale, 'v =', variance, 'n =', noise)
        else:
            print('log-likelihood =', l, 'alpha =', alpha, 'l =', lengthscale, 'v =', variance, 'n =', noise)

    print('log-likelihood =', log_likelihood_base(alpha, beta, lengthscale, variance, noise), flush=True)
    if parser.model in ('reg_diff', 'gf_3random', 'reg_matern3', 'loc_matern2', 'gf_1random', 'gf_diff'):
        print('alpha =', alpha, 'beta =', beta, flush=True)
    else:
        print('alpha =', alpha, flush=True)
    print('l =', lengthscale, 'v =', variance, 'n =', noise, flush=True)

    ll = []
    mses = []
    alpha, beta, lengthscale, variance, noise = np.log(1. + np.exp(alpha)), np.log(1. + np.exp(beta)), np.log(1.+np.exp(lengthscale)), np.log(1.+np.exp(variance)), np.log(1. + np.exp(noise))
    for i in range(1,num_test+1):
        xn1 = joint_x_2(6.25,size).reshape(-1,1)
        yn1 = np.sin(xn1)/xn1 + noisevar*np.random.randn(xn1.shape[0],xn1.shape[1])

        MM = yn1.shape[0] # number of test signals
        I = np.eye(M)
        if parser.model == 'standard':
            B = I
        elif parser.model == 'laplacian':
            B = np.matmul(np.matmul(G.U, np.diag(np.concatenate((np.array([0]), np.sqrt(1./G.e[1:]))))), G.U.T)
        elif parser.model == 'local_averaging':
            B = np.matmul(np.linalg.inv(I + alpha*np.diag(G.d)), I + alpha*np.array(G.W.todense()))
        elif parser.model == 'global_filtering':
            B = np.linalg.inv((I + alpha*np.array(G.L.todense())))
        elif parser.model == 'regularized_laplacian':
            B = np.linalg.inv(I + alpha*np.array(G.L.todense()))
        elif parser.model == 'diffusion':
            B = np.matmul(G.U, np.matmul(np.diag(np.exp(-alpha*G.e)), G.U.T))
        elif parser.model == '1_random_walk': 
            B = (2+alpha)*I + np.array(G.L.todense())
        elif parser.model == '3_random_walk':
            B = np.matmul(G.U, np.matmul(np.diag(((2 + alpha) - G.e)**3), G.U.T))
        elif parser.model == 'cosine':
            B = np.matmul(G.U, np.matmul(np.diag(np.cos(np.pi*G.e/4.)), G.U.T))
        elif parser.model == 'gf_3random':
            B1 = np.linalg.inv((I + alpha*np.array(L1)))
            B2 = np.matmul(U2, np.matmul(np.diag(((2 + beta) - e2)**3), U2.T))
        elif parser.model == 'gf_1random':
            B1 = np.linalg.solve(I + alpha*np.array(L1), I)
            B2 = (2+beta)*I + np.array(L2)
        elif parser.model == 'gf_diff':
            B1 = np.linalg.solve(I + alpha*np.array(L1), I)
            B2 = np.matmul(U2, np.matmul(np.diag(np.exp(-beta*e2)), U2.T))
        elif parser.model == 'reg_diff':
            B1 = np.linalg.inv(I + alpha*np.array(G.L.todense()))
            B2 = np.matmul(G.U, np.matmul(np.diag(np.exp(-beta*G.e)), G.U.T))
            B = B1 + B2
        elif parser.model == 'reg_matern3':
            B1 = np.linalg.inv(I + alpha*np.array(G.L.todense()))
            inv_mat = np.linalg.inv(2*3/beta*I + np.array(G.L.todense()))
            B2 = np.matmul(np.matmul(inv_mat, inv_mat), inv_mat)
            B = B1 + B2
        elif parser.model == 'loc_matern2':
            B1 = np.matmul(np.linalg.inv(I + alpha*np.diag(d1)), I + alpha*np.array(W1))
            inv_mat = np.linalg.inv(2*2/beta*I + np.array(L2))
            B2 = np.matmul(np.matmul(inv_mat, inv_mat), inv_mat)
        elif parser.model == 'matern2':
            inv_mat = np.linalg.inv(2*2/alpha*I + np.array(G.L.todense()))
            B = np.matmul(inv_mat, inv_mat)
        elif parser.model == 'matern3':
            inv_mat = np.linalg.inv(2*3/alpha*I + np.array(G.L.todense()))
            B = np.matmul(np.matmul(inv_mat, inv_mat), inv_mat)
        elif parser.model == 'matern5':
            inv_mat = np.linalg.inv(2*5/alpha*I + np.array(G.L.todense()))
            B = np.matmul(np.matmul(np.matmul(inv_mat, inv_mat), inv_mat), np.matmul(inv_mat, inv_mat))

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

        if parser.model in ('regularized_laplacian', 'diffusion', '1_random_walk', '3_random_walk', 'cosine', 'reg_diff', 'matern2', 'matern3', 'matern5', 'reg_matern3'):
            BB = np.kron(B,np.ones((N,N)))
        elif parser.model in ('gf_3random', 'loc_matern2', 'gf_1random', 'gf_diff'):
            BB1 = np.kron(np.matmul(B1,B1.T),np.ones((N,N)))
            BB2 = np.kron(B2,np.ones((N,N)))
            BB = BB1 + BB2
        else:
            BB = np.kron(np.matmul(B,B.T),np.ones((N,N)))
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