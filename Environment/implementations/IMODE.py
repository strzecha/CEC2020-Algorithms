import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

from implementations.basic_algorithms.improved_MODE import ImprovedMODE
from implementations.basic_algorithms.DE_basic_mutations import (current_to_pbest_with_archive, 
                                                                current_to_pbest_without_archive,
                                                                weighted_rand_to_pbest)

INITIAL_PROBABILITY_LOCAL_SEARCH = 0.1
INITIAL_NP_MIN = 4
INITIAL_ARCHIVE_RATE = 2.6
INITIAL_P_BEST_RATE = 0.1
INITIAL_CR = 0.2
INITIAL_F = 0.2


class IMODE(ImprovedMODE):  
    def __init__(self):
        super().__init__()
        self.name = "IMODE"

    # main methods 
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)

        self.prob_ls = INITIAL_PROBABILITY_LOCAL_SEARCH

        self.NP = 6 * self.D * self.D
        self.NP_min = INITIAL_NP_MIN

        self.A_rate = INITIAL_ARCHIVE_RATE
        self.P_archive = np.array([])
        self.NP_archive = int(self.A_rate * self.NP)

        self.hist_pos = 0
        self.H = 20 * self.D
        self.M_F = np.ones(self.H) * INITIAL_F
        self.M_CR = np.ones(self.H) * INITIAL_CR

        self.NP_init = self.NP

        self.pbest_rate = INITIAL_P_BEST_RATE

        self.limit_all = self.FES_MAX

        self._mutation_strategies = np.array([current_to_pbest_with_archive, 
                                   current_to_pbest_without_archive,
                                   weighted_rand_to_pbest])
    
    def before_start(self):
        super().before_start()

        sorted_index = self._get_sorted_index(self.P)
        self.global_best = deepcopy(self.P[sorted_index[0]])
        self.P = self.P[sorted_index]   

    def local_search(self):
        if np.random.rand() < self.prob_ls:
            LS_FE = min(np.ceil(0.02 * self.FES_MAX), (self.FES_MAX - self.FES)) / 10
    
            options = {'maxiter': LS_FE}
            cons = [{'type': 'ineq', 
             'fun': lambda x, idx=i: self.fun(np.array([x]))[1][0][idx]} for i in range(self.gn)]
            cons.extend([{'type': 'eq', 
                        'fun': lambda x, idx=i: self.fun(np.array([x]))[2][0][idx]} for i in range(self.hn)])
            
            res = minimize(lambda x: self.fun(np.array([x]))[0], 
                            self.global_best.x,
                            method='SLSQP', 
                            bounds=[(mini, maxi) for mini, maxi in zip(self.MIN, self.MAX)],
                            options=options,
                            constraints=cons)
            new_x = self.individual_generic(res.x)
            self._evaluate_individual(new_x)
            new_x.svc = np.sum(self._get_svc(new_x.g, new_x.h))

            if self._is_better(new_x, self.global_best):
                if self.FES + res.nfev <= self.FES_MAX:
                    self.global_best = new_x

                    self.P[-1] = deepcopy(self.global_best)
                    sorted_index = self._get_sorted_index(self.P)
                    self.P = self.P[sorted_index]
                
                self.prob_ls = 0.1
            else:
                self.prob_ls = 0.0001

            if self.FES + res.nfev + 1 <= self.FES_MAX:
                self.FES += (res.nfev + 1)

    def crossover(self):
        if np.random.rand() < 0.3:
            self.O = deepcopy(self.P)
            for i in range(self.NP):
                jrand = np.floor(np.random.rand() * self.D).astype(int)
                for j in range(self.D):
                    if np.random.rand() <= self.P[i].CR or j == jrand:
                        self.O[i].x[j] = self.T[i].x[j]

        else:
            self.O = deepcopy(self.P)
            for i in range(self.NP):
                j = np.random.randint(0, self.D)
                L = 1
                while (np.random.rand() < self.P[i].CR and (L <= self.D)):
                    self.O[i].x[j] = self.T[i].x[j]
                    j = (j + 1) % self.D
                    L += 1

    def selection(self):
        self.better_index = np.zeros(self.NP).astype(bool)
        for i in range(self.NP):
            self.better_index[i] = self._is_better(self.O[i], self.P[i])
        
        if self.hn + self.gn > 0:
            beta = np.zeros(self.NP)
            for i in range(self.NP):
                if self.O[i].svc > 0 and self.P[i].svc > 0:
                    beta[i] = np.nanmax([(self.P[i].svc - self.O[i].svc) / self.P[i].svc, 0]) + \
                        np.nanmax([(self.P[i].objective - self.O[i].objective) / np.abs(self.P[i].objective), 0])
                    
            delta = np.zeros(self.NP)
            beta_max = np.nanmax(beta)
            for i in range(self.NP):
                if self.O[i].svc == 0:
                    objective = self.P[i].objective if self.P[i].objective != 0 else 1
                    delta[i] = beta_max + np.nanmax([(self.P[i].svc - self.O[i].svc) / self.P[i].svc, 0]) + \
                        np.nanmax([(self.P[i].objective - self.O[i].objective) / np.abs(objective), 0])
                    
            self.diff = beta + delta
            self.diff2 = np.maximum(self.diff, 0)
        else:
            self.diff = np.array([])
            self.diff2 = np.array([])
            for i in range(self.NP):
                self.diff = np.append(self.diff, np.abs(self.P[i].objective - self.O[i].objective))
                self.diff2 = np.append(self.diff2, np.maximum(0, (self.P[i].objective - self.O[i].objective)) / np.abs(self.P[i].objective))
        
        # update archive
        self._update_archive(self.P[self.better_index])
        

        # update x and fitx
        self.P[self.better_index] = self.O[self.better_index]

    def after_generate(self):
        super().after_generate()

        sorted_index = self._get_sorted_index(self.P)
        self.P = self.P[sorted_index]

        if self.FES > 0.85 * self.FES_MAX and self.FES < self.FES_MAX:
            self.local_search()
    
    # helpful methods
    def _generate_parameters(self):
        F, CR = self._generate_F_and_CR()

        sorted_index = self._get_sorted_index(self.P)
        self.P = self.P[sorted_index]
        CR = np.sort(CR)

        for i in range(self.NP):
            self.P[i].F = F[i]
            self.P[i].CR = CR[i]

    def _get_p_best_indexes(self):
        sorted_index = self._get_sorted_index(self.P)
        pNP = max(np.round(self.pbest_rate * self.NP), 2)
        randindex = np.floor(np.random.rand(self.NP) * pNP).astype(int)

        return sorted_index[randindex]

    def _update_NP(self):
        NP_new = np.round((((self.NP_min - self.NP_init) / self.FES_MAX) * self.FES) + self.NP_init)
        if self.NP > NP_new:
            sorted_indx = self._get_sorted_index(self.P)
            self.P = np.delete(self.P, sorted_indx[NP_new:], axis=0)
            self.NP = NP_new
            
            self.NP_archive = np.round(self.A_rate * self.NP)
            if self.P_archive.shape[0] > self.NP_archive:
                sorted_index = self._get_sorted_index(self.P_archive)[:self.NP_archive]
                self.P_archive = self.P_archive[sorted_index]
