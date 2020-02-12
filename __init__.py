# pylint: disable=unsubscriptable-object
import numpy as np

def Euc(R, M, S):
    return 0.5*np.sum(np.square(R - M@S))

def g(X, gamma):
    return (1 - np.exp(-np.square(X)/gamma))

def h(X, gamma):
    return (2/gamma)*np.exp(-np.square(X)/gamma)

def dg(X, gamma):
    return (2/gamma)*(X*np.exp(-np.square(X)/gamma))

def gM(M, gamma):
    all_sum = 0
    for i in range(1, M.shape[0]-1):
        all_sum += g(M[i,:] - M[i-1,:], gamma).sum()
        all_sum += g(M[i,:] - M[i+1,:], gamma).sum()
    return all_sum

def hM(M, gamma):
    all_values = np.zeros_like(M)
    all_values[0 ,:] = h(np.zeros(M.shape[1]), gamma)
    all_values[-1,:] = h(np.zeros(M.shape[1]), gamma)
    for i in range(1, M.shape[0]-1):
        all_values[i,:] += h(M[i,:] - M[i-1,:], gamma)
        all_values[i,:] += h(M[i,:] - M[i+1,:], gamma)
    return all_values

def dgM(M, gamma):
    all_values = np.zeros_like(M)
    all_values[0 ,:] = dg(0, gamma)
    all_values[-1,:] = dg(0, gamma)
    for i in range(1, M.shape[0]-1):
        all_values[i,:] += dg(M[i,:] - M[i-1,:], gamma)
        all_values[i,:] += dg(M[i,:] - M[i+1,:], gamma)
    return all_values

def gS(S, gamma, S_size):
    if len(S_size) == 2:
        return 0
    S = S.T.reshape(S_size)
    all_sum = 0
    for i in range(1, S.shape[0]-1):
        all_sum += g(S[i,:,:] - S[i-1,:,:], gamma).sum()
        all_sum += g(S[i,:,:] - S[i+1,:,:], gamma).sum()
    for j in range(1, S.shape[1]-1):
        all_sum += g(S[:,j,:] - S[:,j-1,:], gamma).sum()
        all_sum += g(S[:,j,:] - S[:,j+1,:], gamma).sum()
    return all_sum

def hS(S, gamma, S_size):
    if len(S_size) == 2:
        return 0
    S = S.T.reshape(S_size)
    all_values = np.zeros_like(S)
    all_values[0 ,:,:] = h(0, gamma)
    all_values[-1,:,:] = h(0, gamma)
    all_values[:, 0,:] = h(0, gamma)
    all_values[:,-1,:] = h(0, gamma)
    for i in range(1, S.shape[0]-1):
        all_values[i,:,:] += h(S[i,:,:] - S[i-1,:,:], gamma)
        all_values[i,:,:] += h(S[i,:,:] - S[i+1,:,:], gamma)
    for i in range(1, S.shape[1]-1):
        all_values[:,i,:] += h(S[:,i,:] - S[:,i-1,:], gamma)
        all_values[:,i,:] += h(S[:,i,:] - S[:,i+1,:], gamma)
    return all_values.reshape([-1, all_values.shape[2]]).T

def dgS(S, gamma, S_size):
    if len(S_size) == 2:
        return 0
    S = S.T.reshape(S_size)
    all_values = np.zeros_like(S)
    all_values[0 ,:,:] = dg(0, gamma)
    all_values[-1,:,:] = dg(0, gamma)
    all_values[:, 0,:] = dg(0, gamma)
    all_values[:,-1,:] = dg(0, gamma)
    for i in range(1, S.shape[0]-1):
        all_values[i,:,:] += dg(S[i,:,:] - S[i-1,:,:], gamma)
        all_values[i,:,:] += dg(S[i,:,:] - S[i+1,:,:], gamma)
    for i in range(1, S.shape[1]-1):
        all_values[:,i,:] += dg(S[:,i,:] - S[:,i-1,:], gamma)
        all_values[:,i,:] += dg(S[:,i,:] - S[:,i+1,:], gamma)
    return all_values.reshape([-1, all_values.shape[2]]).T

def initS(R, p):
    S = np.random.rand(p, R.shape[1])
    S = S/S.sum(axis=0)
    return S

def initM(R, p):
    index = list((np.random.rand(p)*R.shape[1]).astype(int))
    M = R.T[index].T
    return M

def D(R, M, S, alpha, betha, gamma_M, gamma_S, S_size):
    return Euc(R, M, S) + alpha*gM(M, gamma_M) + betha*gS(S, gamma_S, S_size)

def newM(R, M, S, alpha, betha, gamma_M, gamma_S):
    M_hM = M*hM(M, gamma_M)
    num = R@S.T + alpha*(M_hM - dgM(M, gamma_S))
    den = M@S@S.T + alpha*M_hM
    return M*num/den

def newS(R, M, S, alpha, betha, gamma_M, gamma_S, org_shape):
    S_hS = S*hS(S, gamma_S, org_shape)
    num = M.T@R + betha*(S_hS - dgS(S, gamma_S, org_shape))
    den = M.T@M@S + betha*S_hS
    return S*num/den

def asignL2(s,m,L2):
    s_m = s - m
    alphas = np.roots([
        np.square(s_m).sum(), 
        2*(m*s_m).sum(),
        np.square(m).sum() - L2**2
    ])
    alpha = np.max(alphas)
    return m + alpha*(s-m)

def asignL1L2(x, L1, L2):
    s = x + (L1 - x.sum())/len(x)
    Z = []
    while True:
        m = [0 if i in Z else L1/(len(x) - len(Z)) for i in range(len(x))]
        m = np.array(m)
        s = asignL2(s, m, L2)
        if np.all(s >= 0):
            return s
        Z = Z + list(np.nonzero(s < 0)[0])
        Z = list(np.unique(Z))
        s = [0 if i in Z else s[i] for i in range(len(s))]
        s = np.array(s)
        c = (s.sum() - L1)/(len(x) - len(Z))
        s = [s[i] if i in Z else s[i] - c for i in range(len(s))]
        s = np.array(s)

def asignSparseness(S, Sp):
    sqrt_N = np.sqrt(S.shape[0])
    L1 = 1
    L2 = L1/(sqrt_N - Sp*(sqrt_N-1))
    for i in range(S.shape[1]):
        S[:,i] = asignL1L2(S[:,i], L1, L2)
    return S

def sparseness(X):
    L1 = np.abs(X).sum(axis=0)
    L2 = np.sqrt(np.square(X).sum(axis=0)) 
    sqrt_N = np.sqrt(X.shape[0])
    return (sqrt_N - L1/L2)/(sqrt_N - 1)

def augmentRM(R, M, delta):
    Ra = np.vstack([R, np.full(R.shape[1], delta)])
    Ma = np.vstack([M, np.full(M.shape[1], delta)])
    return Ra, Ma

def isIterable(iterations):
    is_iterable = True
    try:
        iter(iterations)
    except:
        is_iterable = False
    return is_iterable

def unmix(R, p, Sp, alpha, gamma_M, betha=0, gamma_S=1, delta=1, tol=1e-4, iterations = 10000):
    if len(R.shape) == 3:
        S_shape = np.array([R.shape[0], R.shape[1], p])
        R = R.reshape(-1, R.shape[2]).T
    else:
        S_shape = np.array([p, R.shape[1]])
    M = initM(R, p)
    S = initS(R, p)
    D_old = np.inf
    iterations = iterations if isIterable(iterations) else range(iterations)
    hist = np.empty(len(iterations))
    for i in iterations:
        M = newM(R, M, S, alpha, betha, gamma_M, gamma_S)        
        Ra, Ma = augmentRM(R, M, delta)
        S = newS(Ra, Ma, S, alpha, betha, gamma_M, gamma_S, S_shape)
        S = asignSparseness(S, Sp)        
        D_new = D(R, M, S, alpha, betha, gamma_M, gamma_S, S_shape)
        hist[i] = D_new
        if D_old - D_new <= tol:
            break
        D_old = D_new
    if len(S_shape) == 3:
        S = S.T.reshape(S_shape)
    return M, S, hist[0:i+1]

