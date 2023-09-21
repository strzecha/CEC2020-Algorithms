import numpy as np
from scipy.stats import norm, levy_stable
from copy import deepcopy

from random_generator import random_generator
from implementations.basic_algorithms.MODE import MODEIndividual, MODE
from implementations.basic_algorithms.DE_basic_mutations import current_to_pbest_with_archive
from implementations.basic_algorithms.BCHM import rand_base_1


INITIAL_NP_RATE = 18
INITIAL_NP_MIN = 4
INITIAL_ARCHIVE_RATE = 2.6
INITIAL_PROBABILITY_RATE = 0.25
INITIAL_PROBABILITY_LEVY_MIN = 1e-3
INITIAL_P_BEST_RATE = 0.11
INITIAL_P_TOLERANCE = 0.2
INITIAL_MEMORY_SIZE = 6


class COLSHADEIndividual(MODEIndividual):
    def __init__(self, x=None, objective=None, svc_current=None, svc=None, g=None, h=None, F=None, CR=None):
        super().__init__(x, objective, svc, g, h, F, CR)
        self.svc_current = svc_current


class COLSHADE(MODE):
    def __init__(self):
        self.name = "COLSHADE"
        self.individual_generic = COLSHADEIndividual

    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)

        self.pd_normal = norm(loc=0, scale=0.1)
        self.pd_cauchy = levy_stable(alpha=1.0, beta=0, loc=0, scale=0.1)
        self.pd_levy = levy_stable(alpha=0.5, beta=1, loc=0.01, scale=0.01)

        self.pd_normal.rvs = random_generator.get_normal
        self.pd_cauchy.rvs = random_generator.get_cauchy
        self.pd_levy.rvs = random_generator.custom_levy0

        self.NP_rate = INITIAL_NP_RATE
        self.NP_init = np.ceil(self.NP_rate * self.D).astype(int)
        self.NP = self.NP_init
        self.NP_min = INITIAL_NP_MIN

        self.A_rate = INITIAL_ARCHIVE_RATE
        self.P_archive = np.array([])
        self.NP_archive = 0

        # parameters to mutation
        self.probability_rate = INITIAL_PROBABILITY_RATE
        self.prob_levy_min = INITIAL_PROBABILITY_LEVY_MIN
        self.p_best_rate = INITIAL_P_BEST_RATE
        self._mutation_strategies = np.array([
            self.levy, current_to_pbest_with_archive
        ])

        # parameters to handling constraints
        self.tolerance_final = self.equality_tolerance
        self.FES_tolerance = 0.6 * self.FES_MAX
        self.p_tolerance = INITIAL_P_TOLERANCE

        # memories
        self.H = INITIAL_MEMORY_SIZE
        self.M_F_P = 0.5 * np.ones(self.H)
        self.M_CR_P = 0.5 * np.ones(self.H)
        self.memory_index_P = 0
        self.M_F_L = 0.5 * np.ones(self.H)
        self.M_CR_L = 0.5 * np.ones(self.H)
        self.memory_index_L = 0

    def evaluate_initial_population(self):
        for x in self.P:
            self._evaluate_individual(x)
            x.svc_current = self._get_svc(x, 0)
            self.FES += 1

    def before_start(self):
        super().before_start()

        # initial tolerance
        h = np.array([x.h for x in self.P]).reshape(self.NP, np.maximum(self.hn, 1))
        self.tolerance = np.max(np.abs(h), axis=0)
        self.tolerance = np.maximum(self.tolerance, self.tolerance_final)

        for x in self.P:
            x.svc_current = self._get_svc(x, self.tolerance)
            x.svc = self._get_svc(x, self.tolerance_final)

        sorted_index = self._get_sorted_index(self.P)
        self.P = self.P[sorted_index]

    def repair_boundary_constraints(self):
        for v, x in zip(self.T, self.P):
            v = rand_base_1(v, x, self.D, self.MIN, self.MAX)

    def crossover(self):
        self.O = deepcopy(self.P)

        for u, v in zip(self.O, self.T):
            jrand = np.random.randint(0, self.D)
            for j in range(self.D):
                if np.random.rand() <= u.CR or j == jrand:
                    u.x[j] = v.x[j]

    def evaluate_new_population(self):
        for u in self.O:
            self._evaluate_individual(u)
            u.svc_current = self._get_svc(u, self.tolerance)
            u.svc = self._get_svc(u, self.tolerance_final)
            self.FES += 1

    def selection(self):
        # feasibility tournament
        delta_objective = np.zeros(self.NP)
        delta_svc = np.zeros(self.NP)

        for i in range(self.NP):
            if i == 0:
                if self.O[i].svc_current == self.P[i].svc_current == 0 and self.O[i].objective < self.P[i].objective:
                    delta_objective[i] = self.P[i].objective - self.O[i].objective
                elif self.O[i].svc_current == self.P[i].svc_current == 0 and self.O[i].svc < self.P[i].svc:
                    delta_svc[i] = self.P[i].svc - self.O[i].svc
            else:
                if self.O[i].svc_current == self.P[i].svc_current == 0 and self.O[i].objective < self.P[i].objective:
                    delta_objective[i] = self.P[i].objective - self.O[i].objective
                elif self.O[i].svc_current < self.P[i].svc_current:
                    delta_svc[i] = self.P[i].svc_current - self.O[i].svc_current
                elif self.O[i].svc_current == self.P[i].svc_current == 0 and self.O[i].svc < self.P[i].svc:
                    delta_svc[i] = self.P[i].svc - self.O[i].svc

        delta_objective_max = np.max(delta_objective)
        delta_svc_max = np.max(delta_svc)

        if delta_objective_max > 0:
            delta_objective = delta_objective / delta_objective_max
        if delta_svc_max > 0:
            delta_svc = delta_svc / delta_svc_max

        self.improvement = delta_objective + delta_svc

        for i in range(self.NP):
            if self.improvement[i] > 0:
                self.P_archive = np.append(self.P_archive, self.P[i])
                self.P[i] = self.O[i]

    def after_generate(self):
        self._calculate_improvement()
        self._update_memories()
        self._update_probabilities_of_mutations_strategies()

        self._update_tolerance()
        for i in range(self.NP):
            self.P[i].svc_current = self._get_svc(self.P[i], self.tolerance)

        # sort population
        sorted_index = self._get_sorted_index(self.P)
        self.P = self.P[sorted_index]

        # update NP
        self.NP = np.round(self.NP_min + (1 - (self.FES / self.FES_MAX)) * (self.NP_init - self.NP_min))
        self.P = self.P[:self.NP]

        self._update_archive()

        # find global best
        self.global_best = self.P[0]

        super().after_generate()

    # mutation strategies    
    def levy(self, x, P, P_with_archive, r, pbest):
        levy_rand = self.pd_levy.rvs(size=(1, self.D)).reshape(self.D)
        return self.individual_generic(x.x + x.F * levy_rand * (pbest.x - x.x))

    # helpful methods
    def _calculate_improvement(self):
        self.improvement_pbest = np.zeros(self.NP)
        self.improvement_levy = np.zeros(self.NP)

        for i, x in enumerate(self.P):
            if x.mutation_num == 0: # levy
                self.improvement_levy[i] = self.improvement[i]
            else:
                self.improvement_pbest[i] = self.improvement[i]

    def _update_probabilities_of_mutations_strategies(self):
        if np.any(self.improvement_pbest > 0) or np.any(self.improvement_levy > 0):
            delta_mean = np.sum(self.improvement_levy) / (np.sum(self.improvement_levy) + np.sum(self.improvement_pbest))
            prob_levy = self.probability_rate * self._probability_of_mutation_strategies[0] + (1 - self.probability_rate) * delta_mean
            prob_levy = np.clip(prob_levy, self.prob_levy_min, 1 - self.prob_levy_min)

            self._probability_of_mutation_strategies[0] = prob_levy
            self._probability_of_mutation_strategies[1] = 1 - prob_levy

    def _update_archive(self):
        # update archive
        self.NP_archive_max = np.round(self.NP * self.A_rate)
        self.NP_archive = self.P_archive.shape[0]

        if self.NP_archive > self.NP_archive_max:
            idx = np.random.permutation(self.NP_archive)
            idx = idx[:self.NP_archive - self.NP_archive_max]
            self.P_archive = np.delete(self.P_archive, idx, axis=0)
            self.NP_archive = self.P_archive.shape[0]

    def _check_best_feasibility(self):
        return self.global_best.svc == 0 and not self.is_feasible
    
    def _check_optimum_reached(self):
        return self.global_best.svc == 0 and np.abs(self.global_best.objective - self.optimum) < self.objective_tolerance      

    def _generate_parameters(self):
        for x in self.P:
            r = np.random.randint(0, self.H)
            if x.mutation_num == 0:
                m_cr = self.M_CR_L[r]
                m_f = self.M_F_L[r]
            else:
                m_cr = self.M_CR_P[r]
                m_f = self.M_F_P[r]
            if m_cr != -1:
                x.CR = m_cr + self.pd_normal.rvs()
                
            
            F_crit = np.sqrt((1 - x.CR / 2) / self.NP)

            x.F = 0
            while x.F <= F_crit:
                x.F = m_f + self.pd_cauchy.rvs()
            
            x.F = np.nanmin([x.F, 1])
            x.CR = np.nanmax([np.nanmin([x.CR, 1]), 0])

    def _get_p_best_indexes(self):
        pNP = np.ceil(self.p_best_rate * self.NP).astype(int)
        p_best = np.random.randint(0, pNP, size=self.NP)
        return p_best

    def _get_random_indexes(self, NP2):
        r1 = np.zeros(self.NP).astype(int)
        r2 = np.zeros(self.NP).astype(int)
        
        for i in range(self.NP):
            new_arr = np.delete(np.arange(self.NP), i)
            r1[i] = np.random.choice(new_arr)

            new_arr = np.delete(np.arange(NP2), (i, r1[i]))
            r2[i] = np.random.choice(new_arr)
        
        return r1, np.zeros(self.NP), r2

    def _get_svc(self, x, equality_tolerance):
        return np.sum(np.maximum(x.g, 0)) + np.sum(np.maximum(np.abs(x.h) - equality_tolerance, 0))

    def _update_tolerance(self):
        feasible_individuals = np.sum([1 if x.svc_current == 0 else 0 for x in self.P])

        if (self.FES < self.FES_tolerance):
            for j in range(self.hn):
                if feasible_individuals >= self.p_tolerance * self.NP:
                    decay = (self.tolerance_final / self.tolerance[j]) ** (self.NP / (self.FES_tolerance - self.FES))
                    self.tolerance[j] = self.tolerance[j] * decay
                    self.tolerance[j] = np.maximum(self.tolerance[j], self.tolerance_final)
        else:
            self.tolerance = self.tolerance_final

    def _update_memories(self):
        S_F = np.array([])
        S_CR = np.array([])

        S_F_L = np.array([])
        S_CR_L = np.array([])

        for i, x in enumerate(self.P):
            if self.improvement[i] > 0:
                if x.mutation_num == 0: # levy
                    S_F_L = np.append(S_F_L, x.F)
                    S_CR_L = np.append(S_CR_L, x.CR)
                else:
                    S_F = np.append(S_F, x.F)
                    S_CR = np.append(S_CR, x.CR)           

        if np.size(S_F) > 0:
            nonzero_index = np.where(self.improvement_pbest != 0)
            total_improvement_pbest = self.improvement_pbest[nonzero_index]

            w = total_improvement_pbest / np.sum(total_improvement_pbest)

            if (self.M_CR_P[self.memory_index_P] == -1) or (np.max(S_CR) == 0):
                self.M_CR_P[self.memory_index_P] = -1
            else:
                self.M_CR_P[self.memory_index_P] = self._count_lehmer_mean(S_CR, w)

            self.M_F_P[self.memory_index_P] = self._count_lehmer_mean(S_F, w)
            self.memory_index_P = (self.memory_index_P + 1) % self.H

        if np.size(S_F_L) > 0:
            nonzero_index = np.where(self.improvement_levy != 0)
            total_improvement_levy = self.improvement_levy[nonzero_index]

            w = total_improvement_levy / np.sum(total_improvement_levy)

            if (self.M_CR_L[self.memory_index_L] == -1) or (np.max(S_CR_L) == 0):
                self.M_CR_L[self.memory_index_L] = -1
            else:
                self.M_CR_L[self.memory_index_L] = self._count_lehmer_mean(S_CR_L, w)

            self.M_F_L[self.memory_index_L] = self._count_lehmer_mean(S_F_L, w)
            self.memory_index_L = (self.memory_index_L + 1) % self.H
