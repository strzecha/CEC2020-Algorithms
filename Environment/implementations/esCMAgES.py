import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

from implementations.basic_algorithms.evolutionary_algorithm_with_constraints import EvolutionaryAlgorithmWithConstraints
from implementations.basic_algorithms.BCHM import projection


INITIAL_T = 500
INITIAL_DELTA_MIN = 3
INITIAL_SIGMA_MAX = 100
INITIAL_TOLERANCE_SIGMA = 1e-16
INITIAL_TOLERANCE_FUN = 1e-16
INITIAL_AGE_LIMIT = 300


class esCMAgES(EvolutionaryAlgorithmWithConstraints):
    def __init__(self):
        super().__init__()
        self.name = "esCMAgES"
    
    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)

        self.NP = 4 + np.floor(3 * np.log(self.D)).astype(int)

        self.age = 0
        self.Tt = INITIAL_T
        self.constraint_number = max(self.gn + self.hn, 1)
        self.delta_min = INITIAL_DELTA_MIN
        self.sigmaMAX = INITIAL_SIGMA_MAX

        self.age_limit = INITIAL_AGE_LIMIT
        self.tolerance_sigma = INITIAL_TOLERANCE_SIGMA
        self.tolerance_fun = INITIAL_TOLERANCE_FUN

        self._start_parameters()

    def before_start(self):
        super().before_start()
        self._initialize_tolerance()

        sorted_index = self._get_sorted_population_index(self.tolerance)
        mu_best = np.array([self.P[i].x for i in range(self.NP)])[sorted_index[:self.mu]].T
        self.m = np.dot(mu_best, self.weights)

        best_new = self.P[sorted_index[0]]
        self.old_best_objective = best_new.objective

        self.global_best = deepcopy(best_new)
        self.local_best = deepcopy(best_new)
        self.restart = 0

    def prepare_to_generate(self):
        if self.restart == 1:
            self._reinitialize()
            self.restart = 0

    def genetics_operations(self):
        z = np.random.randn(self.D, self.NP)
        self.R = np.array([self.individual_generic(np.zeros(self.D)) for i in range(self.NP)])
        self.T = np.array([self.individual_generic(np.zeros(self.D)) for i in range(self.NP)])
        self.P = np.array([self.individual_generic(np.zeros(self.D)) for i in range(self.NP)])
        for i in range(self.NP):
            for j in range(self.D):
                self.R[i].x[j] = z[j][i]
            self.T[i].x = np.diag(np.diag(self.C ** 0.5)).dot(self.R[i].x)
            self.P[i].x = self.m + self.sigma * self.T[i].x

        done_repairs = np.zeros(self.NP).astype(bool)
        self.P_new = deepcopy(self.P)
        for i in range(self.NP):
            self.P_new[i] = self.repair_boundary_constraints(deepcopy(self.P[i]))
            done_repairs[i] = np.logical_not(np.array_equal(self.P[i].x, self.P_new[i].x))
            self._evaluate_individual(self.P_new[i])
            self.P_new[i].svc = self._get_svc(self.P_new[i].g, self.P_new[i].h)
            
        self.FES += self.NP

        # Solution Repair
        if (self.hn > 0 and self.D < 20) or self.hn == 0:
            self.maxiter = 4
            pnb = 1
        else:
            self.maxiter = 3000
            pnb = 0.2

        if np.mod(self.t, self.D) == 0:
            for i in range(self.NP):
                if np.random.rand() <= pnb and self.P_new[i].svc > 0:
                    self.P_new[i], FES = self._constraints_repair(self.P_new[i].x)
                    self.P_new[i] = self.repair_boundary_constraints(self.P_new[i])
                    self._evaluate_individual(self.P_new[i])
                    self.FES += FES + 1
                    self.P_new[i].svc = self._get_svc(self.P_new[i].g, self.P_new[i].h)
                    done_repairs[i] = True


        for i in range(self.NP):
            if done_repairs[i]:
                self.T[i].x = (self.P_new[i].x - self.m) / self.sigma
                self.R[i].x = np.dot(np.linalg.inv(np.diag(np.diag(self.C))), self.T[i].x)

    def repair_boundary_constraints(self, x):
        return projection(x, self.D, self.MIN, self.MAX)

    def selection(self):
        self.P = self.P_new

    def after_generate(self):
        sorted_index_final_tolerance = self._get_sorted_population_index(0)
        best_new = self.P[sorted_index_final_tolerance[0]]

        sorted_index = self._get_sorted_population_index(self.tolerance)
        fun_improvement = np.abs(self.old_best_objective - best_new.objective)
        self.old_best_objective = best_new.objective

        if self._is_better(best_new, self.local_best):
            self.local_best = deepcopy(best_new)
            self.age = 0
        else:
            self.age += 1

        if self._is_better(self.local_best, self.global_best):
            self.global_best = deepcopy(self.local_best)
        
        T_sum = np.zeros((self.mu, self.D))
        R_sum = np.zeros((self.mu, self.D))
        
        j = 0
        for i in sorted_index[:self.mu]:
            T_sum[j] = self.T[i].x * self.weights[j]
            R_sum[j] = self.R[i].x * self.weights[j]
            
            j += 1

        T_sum = np.sum(T_sum, axis=0).reshape(self.D, 1)
        R_sum = np.sum(R_sum, axis=0).reshape(self.D, 1)

        self.m = self.m + self.sigma * T_sum.reshape(self.D)

        self._update_p_sigma(R_sum)
        self._update_pc(T_sum)
        self._update_C(sorted_index)                                          
        self._update_sigma()

        if not np.all(np.isfinite(self.C)) or np.max(self.C) > 1e16:
            self.C = np.eye(self.D)

        if self._reinitialize_statements(fun_improvement):
            self.restart = 1
            self.local_best = self.individual_generic(np.zeros(self.D), float('inf'), float('inf'))

        self.t = self.t + 1
        self._update_tolerance()

        super().after_generate()

    # helpful methods
    def _update_pc(self, T_sum):
        h_sigma = np.sum(self.p_sigma ** 2) / ((1 - (1 - self.c_sigma) ** (2 * self.FES / self.NP)) * self.D) < 2 + 4 / (self.D + 1)
        self.pc = (1 - self.cc) * self.pc + h_sigma * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * T_sum

    def _update_p_sigma(self, R_sum):
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * R_sum

    def _update_C(self, sorted_index):
        Tx = np.array([x.x for x in self.T]).T
        self.C = (1 - self.cmu) * self.C + np.dot(self.cmu * (1 - 1 / self.mu_eff) * Tx[:, sorted_index[0:self.mu]], \
                    np.dot(np.diag(self.weights), Tx[:, sorted_index[0:self.mu]].T)) + 1 / self.mu_eff * np.dot(self.pc, self.pc.T)

    def _update_sigma(self):
        self.sigma = np.minimum(self.sigma * np.exp((self.c_sigma / 2) * (np.linalg.norm(self.p_sigma) ** 2 / self.D - 1)),
                          self.sigmaMAX)

    def _reinitialize_statements(self, fun_improvement):
        return (self.sigma < self.tolerance_sigma or 
                self.age > self.age_limit or 
                fun_improvement < self.tolerance_fun) and self.t > self.Tt

    def _get_sorted_population_index(self, E):
        f = np.array([x.objective for x in self.P])
        g = np.array([x.g for x in self.P]).reshape(self.NP, max(self.gn, 1))
        h = np.array([x.h for x in self.P]).reshape(self.NP, max(self.hn, 1))
        G = np.sum(np.maximum(0, g - E), axis=1)
        H = np.sum(np.maximum(0, np.abs(h) - (E + 0.0001)), axis=1)
        svc = G + H
        sorted_index = np.lexsort(np.column_stack((svc, f)).T[::-1])
        return sorted_index

    def _initialize_tolerance(self):
        f = np.array([x.objective for x in self.P])
        svc = np.array([x.svc for x in self.P])
        i = np.lexsort(np.column_stack((svc, f)).T[::-1])
        n = np.ceil(0.9 * self.NP).astype(int)
        self.tolerance = np.median(svc[i[:n]])

        self.initial_tolerance = self.tolerance
        
        self.delta_tolerance = np.maximum(self.delta_min, (-5 - np.log(self.tolerance)) / np.log(0.05))

    def _update_tolerance(self):
        if self.t > 1 and self.t < self.Tt:
            self.tolerance = self.initial_tolerance * ((1 - self.t / self.Tt) ** self.delta_tolerance)
        elif self.t + 1 >= self.Tt:
            self.tolerance = 0

    def _constraints_repair(self, x0):
        options = {'disp': False, 'maxiter': self.maxiter}
    
        cons = [{'type': 'ineq', 
                'fun': lambda x, idx=i: self.fun(np.array([x]))[1][0][idx]} for i in range(self.gn)]
        cons.extend([{'type': 'eq', 'fun': lambda x, idx=i: self.fun(np.array([x]))[2][0][idx]} for i in range(self.hn)])

        bounds = [(mini, maxi) for mini, maxi in zip(self.MIN, self.MAX)]
        func = lambda x: self.fun(np.array([x]))[0]
        
        res = minimize(func, x0, method='SLSQP', options=options,
                        bounds=bounds, constraints=cons)
        new_mutant = self.individual_generic(res.x)
        fes = res.nfev

        return new_mutant, fes

    def _get_svc(self, g, h):
        return super()._get_svc(g, h) / self.constraint_number

    def _reinitialize(self):       
        self.NP = np.floor(1.5 * self.NP)
        self.NP = self.NP.astype(int)

        self._start_parameters()
        self.initialize_population()   
        self.evaluate_initial_population()       
        self._initialize_tolerance()
        
        sorted_index = self._get_sorted_population_index(self.tolerance)
        ParentPop = np.array([self.P[i].x for i in range(self.NP)])[sorted_index[:self.mu]].T
        
        self.m = np.dot(ParentPop, self.weights)

    def _start_parameters(self):
        self.sigma = 1
        self.mu = np.floor(self.NP / 3).astype(int)
        self.weights = np.log(self.mu + 1/2) - np.log(range(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        self.c_sigma = (self.mu_eff + 2) / (self.D + self.mu_eff + 3)
        self.cmu = ((self.D + 2) / 3) * ((2 / (self.mu_eff * (self.D + np.sqrt(2)) ** 2)) + \
                    (1 - 1 / self.mu_eff) * min([1, (2 * self.mu_eff - 1) / ((self.D + 2) ** 2 + self.mu_eff)]))
        self.cc = 4 / (self.D + 4)

        self.p_sigma = np.zeros((self.D, 1))
        self.pc = np.zeros((self.D, 1))
        self.C = np.eye(self.D)

        self.t = 0

    def _is_better(self, x1, x2):
        return (x1.svc == x2.svc == 0 and x1.objective <= x2.objective) or (x1.svc == x2.svc and x1.objective <= x2.objective) or (x1.svc < x2.svc)
