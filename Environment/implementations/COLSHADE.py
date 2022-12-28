import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

alfa = 0.5
beta = 1
gamma = 0.01
delta = 0
mi = 0.25
e_f = 10 ** (-4)
p = 0.11 # proportion of best solutions
H = 6 # size of memory

class COLSHADEIndividual:
    def __init__(self, x):
        self.x = x
        self.objective = 0

    def __repr__(self):
        return f"{self.x}: obj:{self.objective}"

    def __add__(self, individual):
        return COLSHADEIndividual(self.x + individual.x)
    
    def __sub__(self, individual):
        return COLSHADEIndividual(self.x - individual.x)

    def __mul__(self, num):
        return COLSHADEIndividual(self.x * num)

def COLSHADE(dim, MAX_FES, fun):
    
    # 1
    r_N_init = 18
    r_arc = 2.6
    N_init = np.round(dim * r_N_init)
    A = np.round(N_init * r_arc)
    
    # 2
    N_0 = N_init
    N_min = 4
    p_m0 = 0.5 
    p_mG = p_m0
    N_G = N_0

    # 3 initialize memories
    M_CR = 0.5 * np.ones([H, 1])
    M_F = 0.5 * np.ones([H, 1])
    M_CR_L = 0.5 * np.ones([H, 1])
    M_F_L = 0.5 * np.ones([H, 1])

    # 4
    P_0 = [COLSHADEIndividual(np.random.rand(dim)) for _ in range(N_G)]
    #P_0 = [COLSHADEIndividual(np.random.uniform(-10, 10, [1, dim])) for _ in range(N_G)]
    P = P_0

    # 5 initial tolerance
    print(type(P[0].x))
    # 6
    stop = False
    FES = 0
    x = 0

    objs = list()

    while not stop:
        x += 1
        # 7
        S_CR = np.array([])
        S_F = np.array([])
        S_CR_L = np.array([])
        S_F_L = np.array([])
        # 8
        delta_f = np.array([])
        delta_f_L = np.array([])

        for i in range(N_G):
            P[i].objective = evaluate(P[i], fun)

        pbest = get_pbest(P)

        objs.append(pbest[0].objective)

        
        # 9
        for i in range(N_G):
            new_P = P[:]
            # 10
            l = np.random.rand()
            
            # 11
            if l <= p_mG:
                # 12
                CR_i, F_i = generate_parameters(M_CR_L, M_F_L)
                # 13
                u_i_g = current_to_pbest(P[i], CR_i, F_i, P, pbest)
            # 14
            else:
                # 15
                CR_i, F_i = generate_parameters(M_CR, M_F)
                # 16
                u_i_g = levy(P[i], CR_i, F_i, P, pbest, N_G, dim)

            u_i_g.objective = evaluate(u_i_g, fun)
            # 18
            if u_i_g.objective < P[i].objective:
                #print("lepszy")
                # 19
                new_P[i] = u_i_g
                # 20
                # A.append(P[i])
                # 21
                if l <= p_mG:
                    # 22
                    S_CR_L = np.append(S_CR_L, CR_i)
                    S_F_L = np.append(S_F_L, F_i)
                    # 23
                    # imporve delta_f_L
                else:
                    # 25
                    S_CR = np.append(S_CR, CR_i)
                    S_F = np.append(S_F, F_i)
                    # 26
                    # imporve delta_F

        P = new_P[:]
      
        FES += N_G

        M_CR, M_F = update_memories(M_CR, M_F, S_CR, S_F, delta_f)
        M_CR_L, M_F = update_memories(M_CR, M_F, S_CR_L, S_F, delta_f_L)
        p_mG = update_probability(delta_f, delta_f_L, p_mG, mi)
        # e_G = 
        N_G = int(np.round(((N_min - N_init) / MAX_FES) * FES + N_init))
        
        if FES >= MAX_FES:
            break

    plt.plot(range(len(objs)), objs)
    plt.show()

####### handling constraints #######
def svc(x, e_G):
    I_x = 0 #TODO

def L2(S, w):
    pass

def update_memories(M_CR, M_F, S_CR, S_F, delta_f):
    return M_CR, M_F # TODO

def update_probability(delta_f, delta_f_L, p_mG, mi):
    if np.size(delta_f_L) == 0 and np.size(delta_f) == 0:
        return p_mG
    else:
        p_mG_s = np.sum(delta_f_L) / (np.sum(delta_f_L) + np.sum(delta_f))
        return mi * p_mG + (1 - mi) * p_mG_s

def generate_parameters(M_CR, M_F):
    ri = np.random.randint(1, np.size(M_CR))
    CR_i = np.random.normal(M_CR[ri], 0.1) if M_CR[ri] > 0 else 0
    F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()

    return CR_i, F_i

def current_to_pbest(x_i, CR_i, F_i, P, pbest):
    x_pbest = np.random.choice(pbest)
    x_r1 = x_r2 = None
    
    while x_r1 == x_r2:
        x_r1, x_r2 = np.random.choice(P, 2)

    return x_i + (x_pbest - x_i) * F_i + (x_r1 - x_r2) * F_i

def levy(x_i, CR_i, F_i, P, pbest, N_G, dim):
    F_crit = np.sqrt((1 - CR_i/2) / N_G)
    F_i = np.max([F_crit, F_i])
    jrand = np.random.randint(1, dim+1)
    x_pbest = np.random.choice(pbest)
    u_i = copy.deepcopy(x_i)
    for j in range(dim):
        if np.random.rand() <= CR_i or j == jrand:
            F_levy = F_i * sp_stats.levy_stable.rvs(alfa, beta, gamma, gamma + delta)
            u_i.x[j] = x_i.x[j] + F_levy * (x_pbest.x[j] - x_i.x[j])
        else:
            u_i.x[j] = x_i.x[j]
    return u_i
    

def get_pbest(P):
    best = sorted(P, key=lambda x: x.objective)
    ind = int(np.round(p * np.size(P)))
    return best[:ind]

def evaluate(x, fun):
    return fun(x.x)

