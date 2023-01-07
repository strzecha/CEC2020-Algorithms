import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from implementations.evolutionary_algorithm import EvolutionaryAlgorithm, Individual

alfa = 0.5
beta = 1
gamma = 0.01
delta = 0
mi = 0.25
e_f = 10 ** (-4)
p = 0.11 # proportion of best solutions
H = 6 # size of memory

class COLSHADEIndividual(Individual):
    def __init__(self, x, F=0, CR=0):
        super().__init__(x)
        self.F = F
        self.CR = CR
        self.h = None
        self.g = None

    def is_efeasible(self, e):
        for i in range(np.size(self.h)):
            if self.h[i] - e[i] > 0:
                return False
        return True

    def __repr__(self):
        return f"{self.x}, {self.objective}, {self.h}, {self.g}"

    def __add__(self, individual):
        return COLSHADEIndividual(self.x + individual.x, self.F, self.CR)

    def __sub__(self, individual):
        return COLSHADEIndividual(self.x - individual.x, self.F, self.CR)

    def __mul__(self, num):
        return COLSHADEIndividual(self.x * num, self.F, self.CR)

class COLSHADE(EvolutionaryAlgorithm):
    def __init__(self, alpha=0.5, beta=1, gamma=0.01, delta=0, mi=0.25, p=0.11, 
                memory_size=6, r_NP_init=18, r_arc=2.6, NP_min=4, p_m=0.5, p_m_min=10**(-3), p_f=0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mi = mi
        self.p = p # proportion of best solutions
        self.p_f = p_f # proportion of e-feasible solutions
        self.memory_size = memory_size # size of memory
        self.r_NP_init = r_NP_init
        self.r_arc = r_arc
        self.NP_min = NP_min
        self.p_m = p_m
        self.p_m_min = p_m_min

    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.FESe = 0.6 * self.MAX_FES

        self.NP_init = np.round(self.D * self.r_NP_init)
        self.NP = self.NP_init

        self.archive_size = np.round(self.NP_init * self.r_arc)
        self.archive = np.array([])

        self.M_CR = 0.5 * np.ones([self.memory_size, 1])
        self.M_F = 0.5 * np.ones([self.memory_size, 1])
        self.M_CR_L = 0.5 * np.ones([self.memory_size, 1])
        self.M_F_L = 0.5 * np.ones([self.memory_size, 1])

    def initialize_population(self):
         self.P = np.array([COLSHADEIndividual(np.random.uniform(self.MIN, self.MAX, self.D)) for i in range(self.NP)])

    def evaluate_initial_population(self):
        for i in range(self.NP):
            objective, h, g = self.fun(self.P[i].x)
            self.P[i].objective = objective
            self.P[i].h = h
            self.P[i].g = g
        self.FES += self.NP

        

    def get_pbest(self):
        best = sorted(self.P, key=lambda x: self.svc(x, self.e_t))

        for i in range(self.NP):
            if self.svc(best[i], self.e_t) > 0:
                end = i
                break

        best[:i] = sorted(best[:i], key=lambda x: x.objective)

        ind = int(np.round(self.p * np.size(self.P)))
        ind = np.max([ind, 1])
        self.pbest = best[:ind]
        self.global_best = self.pbest[0]

    def before_start(self):
        self.e_t = np.amax(abs(np.array([self.P[i].h for i in range(self.NP)])), axis=0)
        self.e_f = np.array(np.size(self.e_t) * [ 10 ** (-4) ])
        self.get_pbest()
        self.FESs.append(self.FES)
        self.bests_values.append(self.global_best.objective)

    def prepare_to_generate_population(self):
        self.S_CR = np.array([])
        self.S_F = np.array([])
        self.S_CR_L = np.array([])
        self.S_F_L = np.array([])
        # 8
        self.delta_f = np.array([])
        self.delta_f_L = np.array([])

        self.l = np.random.rand()

        self.old_P = copy.deepcopy(self.P)

    def mutation(self):
        self.T = np.array([])

        for i in range(self.NP):
            if self.l <= self.p_m:
                # 12
                self.P[i].CR, self.P[i].F = self.generate_parameters(self.M_CR_L, self.M_F_L)
                # 13
                v = current_to_pbest(self.P[i], self.P[i].CR, self.P[i].F, self.P, self.pbest, self.archive)
            # 14
            else:
                # 15
                self.P[i].CR, self.P[i].F = self.generate_parameters(self.M_CR, self.M_F)
                # 16
                v = levy(self.P[i], self.P[i].CR, self.P[i].F, self.P, self.pbest, self.NP)

            self.T = np.append(self.T, v)

    def crossover(self):
        self.O = np.array([])
        

        for i in range(self.NP):
            j_rand = np.random.randint(0, self.D)
            v = self.T[i]
            x = self.P[i]
            u = np.array([])
            for j in range(self.D):
                if np.random.rand() < v.CR or j == j_rand:
                    u = np.append(u, v.x[j])
                else:
                    u = np.append(u, x.x[j])

                # handling boundary constraints
                while u[j] > self.MAX[j]:
                    r = np.random.rand() / 10
                    u[j] = (1 - r) * self.MAX[j] + r * x.x[j]
                while u[j] < self.MIN[j]:
                    r = np.random.rand() / 10
                    u[j] = (1 - r) * self.MIN[j] + r * x.x[j]
            
            self.O = np.append(self.O, COLSHADEIndividual(u, x.F, x.CR))

    def evaluate_new_population(self):
        for i in range(self.NP):
            objective, h, g = self.fun(self.O[i].x)
            self.O[i].objective = objective
            self.O[i].h = h
            self.O[i].g = g

        self.FES += self.NP
                
    def selection(self):
        new_P = self.feasibility_binary_tournament()
        self.P = new_P

    def feasibility_binary_tournament(self):
        delta_f = np.zeros([self.NP, 1])
        delta_svc = np.zeros([self.NP, 1])

        new_P = np.array([])

        for i in range(self.NP):
            u = self.O[i]
            x = self.old_P[i]

            svc_u = self.svc(u, self.e_t)
            svc_x = self.svc(x, self.e_t)
            svc_u_abs = self.svc(u, self.e_f)
            svc_x_abs = self.svc(x, self.e_f)

            if svc_u < svc_x:
                delta_svc[i] = svc_x - svc_u # feasibility
            elif svc_u == svc_x == 0 and u.objective < x.objective:
                delta_f[i] = x.objective - u.objective # optimality
            elif self.svc(u, self.e_t) == self.svc(x, self.e_t) == 0 and svc_u_abs < svc_x_abs:
                delta_svc[i] = svc_x_abs - svc_u_abs # feasibility

        delta_f_max = np.max(delta_f)
        delta_svc_max = np.max(delta_svc)

        if delta_f_max > 0:
            delta_f = delta_f / delta_f_max
        if delta_svc_max > 0:
            delta_svc = delta_svc / delta_svc_max

        delta_f = delta_f + delta_svc

        for i in range(self.NP):
            u = self.O[i]
            x = self.P[i]
            if delta_f[i] > 0:
                new_P = np.append(new_P, u)
                self.update_archive(x)
                if self.l <= self.p_m:
                    self.S_CR_L = np.append(self.S_CR_L, u.CR)
                    self.S_F_L = np.append(self.S_F_L, u.F)
                    self.delta_f_L = delta_f
                else:
                    self.S_CR = np.append(self.S_CR, u.CR)
                    self.S_F = np.append(self.S_F, u.F)
                    self.delta_f = delta_f
            else:
                new_P = np.append(new_P, x)

        return new_P       

    def after_generate(self):
        if self.FES >= self.MAX_FES:
            self.stop = True

        self.get_pbest()
        self.FESs.append(self.FES)
        self.bests_values.append(self.global_best.objective)

        self.M_CR, self.M_F = self.update_memories(self.M_CR, self.M_F, self.S_CR, self.S_F, self.delta_f)
        self.M_CR_L, self.M_F_L = self.update_memories(self.M_CR_L, self.M_F_L, self.S_CR_L, self.S_F_L, self.delta_f_L)
        self.update_probability()
        self.update_tolerance()
        self.NP = self.LPSR(self.NP_min, self.NP_init, self.MAX_FES, self.FES)
        self.archive_size = int(self.NP  * self.r_arc)

    def update_probability(self):
        if not (np.size(self.delta_f_L) == 0 and np.size(self.delta_f) == 0):
            total = (np.sum(self.delta_f_L) + np.sum(self.delta_f))
            if total != 0:
                self.p_m_s = np.sum(self.delta_f_L) / total
                self.p_m = self.mi * self.p_m + (1 - self.mi) * self.p_m_s
                self.p_m = np.min([np.max([self.p_m, self.p_m_min]), 1 - self.p_m_min])

####### handling constraints #######
    def svc(self, x, e):
        E_h = np.sum([max(x.h[i] - e[i], 0) for i in range(len(x.h))])
        E_g = np.sum([max(x.g[i], 0) for i in range(len(x.g))])

        return E_g + E_h

    def L2(self, S, delta_f):
        w = [delta_f[s] / sum(delta_f[i] for i in range(len(delta_f))) for s in range(len(S))]
        total = np.sum([w[s] * S[s] for s in range(len(S))])
        if total == 0:
            total = 10 ** (-4)
        return np.sum([w[s] * S[s] ** 2 for s in range(len(S))]) / total

    def update_tolerance(self):
        if self.is_efeasible():
            self.e_t = self.e_t * (self.e_f / self.e_t) ** (self.NP / (self.FESe - self.FES))
            self.e_t = np.amax([self.e_t, self.e_f], axis=0)

    def is_efeasible(self):
        total = 0

        for i in range(self.NP):
            if self.P[i].is_efeasible(self.e_t):
                total += 1

        return total >= self.p_f * self.NP

    def update_memories(self, M_CR, M_F, S_CR, S_F, delta_f):
        M_CR_k = self.L2(S_CR, delta_f) if np.size(S_CR) > 0 else M_CR[-1]
        M_F_k = self.L2(S_F, delta_f) if np.size(S_F) > 0 else M_F[-1]
        M_CR = np.delete(np.append(M_CR, M_CR_k), 0)
        M_F = np.delete(np.append(M_F, M_F_k), 0)
        return M_CR, M_F

    def generate_parameters(self, M_CR, M_F):
        ri = np.random.randint(0, np.size(M_CR))
        CR_i = np.random.normal(M_CR[ri], 0.1) if M_CR[ri] > 0 else -1
        F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()

        F_crit = np.sqrt((1 - CR_i/2) / self.NP)
        F_i = np.min([np.max([F_crit, F_i]), 1])

        return CR_i, F_i

    def update_archive(self, x):
        self.archive = np.append(self.archive, x)

        while np.size(self.archive) > self.archive_size:
            index = np.random.randint(0, np.size(self.archive_size))
            self.archive = np.delete(self.archive, index)

def current_to_pbest(x_i, CR_i, F_i, P, pbest, archive):
    x_pbest = np.random.choice(pbest)
    r1 = r2 = None

    P_A = np.append(P, archive)
    
    while r1 == r2:
        r1, r2 = np.random.randint(0, np.size(P_A), 2)

    x_r1 = P_A[r1]
    x_r2 = P_A[r2]

    return x_i + (x_pbest - x_i) * F_i + (x_r1 - x_r2) * F_i

def levy(x_i, CR_i, F_i, P, pbest, NP):
    
    x_pbest = np.random.choice(pbest)
    F_levy = F_i * sp_stats.levy_stable.rvs(alfa, beta, gamma, gamma + delta)

    return x_i + (x_pbest - x_i) * F_levy
    



