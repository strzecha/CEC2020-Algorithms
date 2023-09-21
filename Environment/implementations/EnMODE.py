import numpy as np
from copy import deepcopy

from implementations.basic_algorithms.improved_MODE import ImprovedMODE
from implementations.basic_algorithms.DE_basic_mutations import (current_to_pbest_with_archive, 
                                                                current_to_pbest_without_archive)


INITIAL_NP = 200
INITIAL_NP_MIN = 4
INITIAL_ARCHIVE_RATE = 1.4
INITIAL_MEMORY_SIZE = 5
INITIAL_P_BEST_RATE = 0.1
INITIAL_CR = 0.2
INITIAL_F = 0.5

class EnMODE(ImprovedMODE): 
    def __init__(self):
        super().__init__()
        self.name = "EnMODE"
    
    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)

        self.NP = INITIAL_NP
        self.NP_init = self.NP
        self.NP_min = INITIAL_NP_MIN

        self.A_rate = INITIAL_ARCHIVE_RATE
        self.NP_archive = int(self.A_rate * self.NP)

        # memories
        self.H = INITIAL_MEMORY_SIZE
        self.M_F = INITIAL_F * np.ones(self.H)
        self.M_CR = INITIAL_CR * np.ones(self.H)
        self.hist_pos = 0
        
        # parameters to mutation
        self._mutation_strategies = np.array([
            current_to_pbest_with_archive, current_to_pbest_without_archive
            ])
        self.pbest_rate = INITIAL_P_BEST_RATE

        # parameter to adapt probability od mutations
        self.limit_all = self.FES_MAX * 0.75

    def crossover(self):
        self.O = deepcopy(self.P)
        for u, v in zip(self.O, self.T):
            jrand = np.floor(np.random.rand() * self.D)
            for j in range(self.D):
                if np.random.rand() <= u.CR or j == jrand:
                    u.x[j] = v.x[j]

    def selection(self):
        beta = np.zeros(self.NP)
        for i, (x, u) in enumerate(zip(self.P, self.O)):
            if u.svc > 0 and x.svc > 0:
                beta[i] = np.nanmax([(x.svc - u.svc) / x.svc, 0]) + \
                    np.nanmax([(x.objective - u.objective) / np.abs(x.objective), 0])
                
        delta = np.zeros(self.NP)
        beta_max = np.nanmax(beta)
        for i, (x, u) in enumerate(zip(self.P, self.O)):
            if u.svc == 0:
                objective = x.objective if x.objective != 0 else 1
                delta[i] = beta_max + np.nanmax([(x.svc - u.svc) / x.svc, 0]) + \
                    np.nanmax([(x.objective - u.objective) / np.abs(objective), 0])
                
        self.diff = beta + delta
        self.diff2 = self.diff

        self.better_index = np.zeros(self.NP).astype(bool)
        for i in range(self.NP):
            self.better_index[i] = self._is_better(self.O[i], self.P[i])
            if self.better_index[i]:
                self.P[i] = self.O[i]
    
    # helpful methods
    def _get_p_best_indexes(self):
        objectives = np.array([X.objective for X in self.P])
        sorted_index = np.argsort(objectives)
        pNP = max(np.round(self.pbest_rate * self.NP), 2)
        randindex = np.floor(np.random.rand(self.NP) * pNP).astype(int)

        return sorted_index[randindex]

    def _generate_parameters(self):
        F, CR = self._generate_F_and_CR()

        for i in range(self.NP):
            self.P[i].CR = CR[i]
            self.P[i].F = F[i]
    
    def _update_NP(self):
        NP_new = np.round(((self.NP_min - self.NP_init) / self.FES_MAX) * self.FES + self.NP_init)
    
        if self.NP > NP_new:
            sorted_indx = self._get_sorted_index(self.P)
            self.P = np.delete(self.P, sorted_indx[NP_new:], axis=0)
            self.NP = NP_new
