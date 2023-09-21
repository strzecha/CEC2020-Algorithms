import numpy as np
from copy import deepcopy

from implementations.basic_algorithms.MODE import MODE, MODEIndividual
from implementations.basic_algorithms.BCHM import midpoint_target

INITIAL_C = 0.05
INITIAL_P = 0.05
INITIAL_NP_MIN = 12
INITIAL_PROBABILITIES = np.array([0.85, 0.05, 0.05, 0.05])
INITIAL_PAIRS = np.array([{"F" : 0.1, "CR" : 0.2}, 
                                 {"F" : 1.0, "CR" : 0.1},
                                 {"F" : 0.5, "CR" : 0.9},
                                 {"F" : 1.0, "CR" : 0.9}])


class AGSKIndividual(MODEIndividual):
    def __init__(self, x=None, objective=None, svc=None, g=None, h=None, F=None, CR=None, mutation_num=None, 
                 k=None, D_j=None, D_s=None):
        super().__init__(x, objective, svc, g, h, F, CR, mutation_num)
        self.k = k
        self.D_j = D_j
        self.D_s = D_s


class AGSK(MODE):
    def __init__(self):
        self.individual_generic = AGSKIndividual
        self.name = "AGSK"

    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)

        self.NP = 20 * self.D
        self.NP_init = self.NP
        self.NP_min = INITIAL_NP_MIN

        self.pairs_s = INITIAL_PAIRS

        # parameters to 
        self.c = INITIAL_C
        self.p = INITIAL_P

        self._mutation_strategies = np.ones(4)
        
    def before_start(self):
        super().before_start()
        self.global_best = deepcopy(self.P[0])
        self._probability_of_mutation_strategies = INITIAL_PROBABILITIES

        k_rand = np.random.rand(self.NP)

        for i, x in enumerate(self.P):
            if k_rand[i] < 0.5:
                x.k = np.random.rand()
            else:
                x.k = np.ceil(20 * np.random.rand())

    def prepare_to_generate(self):
        super().prepare_to_generate()

        for x in self.P:
            x.D_j = np.ceil((self.D) * (1 - self.FES / self.FES_MAX) ** x.k).astype(int)
            x.D_s = self.D - x.D_j

    def mutation(self):
        self._assign_mutations_strategies()
        self._generate_parameters()
        sorted_index = self._get_sorted_index(self.P)
        rj1, rj2, rj3 = self._get_random_indexes_junior(sorted_index)
        rbest, rm, rworst = self._get_random_indexes_senior(sorted_index)

        self.T = np.array([self.individual_generic(np.zeros(self.D)) for i in range(self.NP)])


        for i in range(self.NP):
            for j in range(0, self.P[i].D_j):
                if self.P[i].objective > self.P[rj3[i]].objective:
                    self.T[i].x[j] = self.P[i].x[j] + self.P[i].F * (self.P[rj1[i]].x[j] - self.P[rj2[i]].x[j] + self.P[rj3[i]].x[j] - self.P[i].x[j])
                else:
                    self.T[i].x[j] = self.P[i].x[j] + self.P[i].F * (self.P[rj1[i]].x[j] - self.P[rj2[i]].x[j] + self.P[i].x[j] - self.P[rj3[i]].x[j])

            for j in range(self.P[i].D_j, self.P[i].D_s):
                if self.P[i].objective > self.P[rm[i]].objective:
                    self.T[i].x = self.P[i].x + self.P[i].F * (self.P[rbest[i]].x - self.P[i].x + self.P[rm[i]].x - self.P[rworst[i]].x)
                else:
                    self.T[i].x = self.P[i].x + self.P[i].F * (self.P[rbest[i]].x - self.P[rm[i]].x + self.P[i].x - self.P[rworst[i]].x)

    def repair_boundary_constraints(self):
        for v, x in zip(self.T, self.P):
            v = midpoint_target(v, x, self.D, self.MIN, self.MAX)
        
    def crossover(self):
        self.O = deepcopy(self.P)

        for u, v in zip(self.O, self.T):
            for j in range(self.D):
                if np.random.rand() <= u.CR:
                    u.x[j] = v.x[j]

    def selection(self):
        self.improvement_value = np.zeros(self.NP)
        self.improvement_index = np.zeros(self.NP).astype(bool)

        for i, (x, u) in enumerate(zip(self.P, self.O)):
            if u.svc == 0 and x.svc == 0:
                self.improvement_value[i] = abs(x.objective - u.objective)
            elif u.svc == 0 and x.svc > 0:
                self.improvement_value[i] = abs(x.svc - u.svc)
            elif u.svc > 0 and x.svc > 0:
                self.improvement_value[i] = abs(x.svc - u.svc)

            self.improvement_index[i] = self._is_better(u, x)
            if self.improvement_index[i]:
                self.P[i] = self.O[i]
    
    def after_generate(self):
        sorted_index = self._get_sorted_index(self.P)
        self.global_best = self.P[sorted_index[0]]
        
        self._calculate_improvement()
        self._update_probabilities_of_mutations_strategies()

        self._update_NP()
        super().after_generate()

    # helpful methods
    def _generate_parameters(self):
        for x in self.P:
            x.F = self.pairs_s[x.mutation_num]['F']
            x.CR = self.pairs_s[x.mutation_num]['CR']

    def _calculate_improvement(self):
        self.improvement = [0] * 4
        
        for i, x in enumerate(self.P):
            if self.improvement_index[i]:
                self.improvement[x.mutation_num] += self.improvement_value[i]
        
        self.improvement_rate = np.zeros(self._mutation_strategies_num)
        if np.sum(self.improvement) != 0:
            for i in range(self._mutation_strategies_num):
                self.improvement_rate[i] = max(self.improvement[i] / sum(self.improvement), 0.05)
            sorted_index = np.argsort(self.improvement_rate)
            self.improvement_rate[sorted_index[-1]] = 1 - np.sum(self.improvement_rate[sorted_index[:-1]])
        else:
            self.improvement_rate = np.ones(self._mutation_strategies_num) / self._mutation_strategies_num
    
    def _get_random_indexes_senior(self, sorted_index):
        rbest = sorted_index[:np.round(self.NP*self.p)]
        rbest_rand = np.floor(len(rbest) * np.random.rand(self.NP)).astype(int)
        rbest = rbest[rbest_rand]
        
        rm = sorted_index[np.round(self.NP*self.p):np.round(self.NP*(1 - self.p))]
        rm_rand = np.floor(len(rm) * np.random.rand(self.NP)).astype(int)
        rm = rm[rm_rand]
        
        rworst = sorted_index[np.round(self.NP*(1 - self.p)):]
        rworst_rand = np.floor(len(rworst) * np.random.rand(self.NP)).astype(int)
        rworst = rworst[rworst_rand]
        
        return rbest, rm, rworst

    def _get_random_indexes_junior(self, sorted_index):
        R1 = np.zeros(self.NP).astype(int)
        R2 = np.zeros(self.NP).astype(int)
        
        for i in range(self.NP):
            ind = np.where(sorted_index == i)[0][0]
            if ind == 0: # best
                R1[i] = sorted_index[1]
                R2[i] = sorted_index[2]
            elif ind == self.NP-1: # worst
                R1[i] = sorted_index[self.NP-3]
                R2[i] = sorted_index[self.NP-2]
            else:
                R1[i] = sorted_index[ind-1]
                R2[i] = sorted_index[ind+1]
        
        R0 = np.arange(self.NP).astype(int)
        R3 = np.floor(np.random.rand(self.NP) * self.NP).astype(int)
        wrong_indexes = np.where((R3 == R2) | (R3 == R1) | (R3 == R0))[0]
        for idx in wrong_indexes:
            new_arr = np.delete(np.arange(self.NP), (R2[idx], R1[idx], R0[idx]))
            R3[idx] = np.random.choice(new_arr)
        
        return R1, R2, R3

    def _update_NP(self):
        NP_new = np.round((self.NP_min - self.NP_init) * ((self.FES / self.FES_MAX) ** (1 - self.FES / self.FES_MAX)) + self.NP_init)

        if self.NP > NP_new:
            reduction_ind_num = self.NP - NP_new
            if self.NP - reduction_ind_num <  self.NP_min:
                reduction_ind_num = self.NP - self.NP_min
                
            self.NP = self.NP - reduction_ind_num
            sorted_index = self._get_sorted_index(self.P)
            worst_ind = sorted_index[-reduction_ind_num:]
            self.P = np.delete(self.P, worst_ind, axis=0)

    def _update_probabilities_of_mutations_strategies(self):
        if self.FES >= 0.1 * self.FES_MAX: 
            self._probability_of_mutation_strategies = (1 - self.c) * np.array(self._probability_of_mutation_strategies) + self.c * np.array(self.improvement_rate)
            self._probability_of_mutation_strategies /= sum(self._probability_of_mutation_strategies)
