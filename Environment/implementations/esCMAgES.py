import numpy as np
import matplotlib.pyplot as plt
import copy

class CMAESIndividual:
    def __init__(self, x):
        self.x = x
        self.objective = 0

    def __repr__(self):
        return f"{self.x}: obj:{self.objective}"

    def __add__(self, individual):
        return CMAESIndividual(self.x + individual.x)
    
    def __sub__(self, individual):
        return CMAESIndividual(self.x - individual.x)

    def __mul__(self, num):
        return CMAESIndividual(self.x * num)

    def __lt__(self, individual):
        return list(self.x) < list(individual.x)

def CMAES(MAXFES, fun, dim):
    #1
    C = np.identity(dim)
    p_c = np.zeros((1, dim))
    p_sigma = np.zeros((1, dim))
    m_t = np.array([0] * dim)
    sigma_t = 0.3
    NP = 4 + int(np.floor(3 * np.log(dim)))
    mi = NP // 2
    c_sigma = dim / 3
    c_c = dim / 4
    d_sigma = 0.9
    c_1 = 2 / (dim ** 2)
    c_mi = mi / (dim ** 2)
    invsqrtC = copy.deepcopy(C)

    chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    objs = list()
    #2
    FES = 0
    stop = False
    while not stop:
        # 3
        d_t = list()
        P = list()
        for i in range(NP):
            d = np.random.multivariate_normal(np.zeros(dim), C)
            d_t.append(d)
            q = m_t + sigma_t * d
            P.append(CMAESIndividual(q))
        print("d_t", d_t)
        # 4
        for i in range(NP):
            P[i].objective = evaluate(P[i], fun)
        FES += NP

        P = sorted(P, key=lambda x: x.objective)

        P = P[:mi] # mi best solutions
        print("BEST:", P[0])
        objs.append(P[0].objective)
        # 5

        delta_t = 1 / mi * np.sum([d for d in d_t], axis=0)
        print("delta_t", delta_t)
        # 6
        m_t = m_t + sigma_t * delta_t

        print("P", P)
        print("m_t", m_t)

        # 7
        p_sigma = (1 - c_sigma) * p_sigma + invsqrtC * np.sqrt(1 - (1 - c_sigma) ** 2) * np.sqrt(mi) @ delta_t
        print("p_sigma", p_sigma)

        # 8
        a = (1 - c_c) * p_c
        b = np.sqrt(1 - (1 - c_c) ** 2) * np.sqrt(mi) * delta_t
        p_c = a + b
        print("p_c", p_c)

        # 9
        sigma_t = sigma_t * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma / chiN - 1)))

        # 10
        
        D_I = np.zeros([dim, dim])
        for i in range(mi):
            d_tt = d_t[i][np.newaxis]
            D_I += np.matmul(d_tt.T, d_tt)
        print("D_I", D_I)
        C = (1 - c_1 - c_mi) * C + c_1 * p_c  * np.transpose(p_c)  + c_mi * np.sum(d)
        C1 = np.triu(C) + np.triu(C, 1).T
        [B, D] = np.linalg.eig(C1)
        D = np.sqrt(np.diag(D))
        invsqrtC = B * np.diag(D ** -1) * B.T
        print("C", C)

        if FES >= MAXFES:
            plt.plot(range(len(objs)), objs)
            plt.show()
            break

def evaluate(x, fun):
    return fun(x.x)

def esCMAgES(MAXFES, fun, dim):
    
    lambda_0 = 4 + np.floor(3 * np.log(dim))
    mi = np.floor(lambda_0 / 2)
    sigma_0 = 1
    P_sigma = 0
    P_c = 0
    C_0 = 0
    S_1 = np.array([])
    FES = 0

    while FES <= MAXFES:
        for i in range(lambda_0):
            y = list()
            for j in range(dim):
                y.append(l_j + (u_j - l_j) * np.random.uniform(0, 1))
            
