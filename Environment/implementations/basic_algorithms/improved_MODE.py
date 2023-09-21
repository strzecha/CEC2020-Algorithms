import numpy as np

from implementations.basic_algorithms.MODE import MODE
from implementations.basic_algorithms.BCHM import rand_base_2, midpoint_target, min_max_reflection


class ImprovedMODE(MODE):
    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.W = 50
        self.t = 0

    def evaluate_initial_population(self):
        self.violation_of_each_constraint = np.zeros((self.NP, max(self.gn + self.hn, 1)))
        for i, x in enumerate(self.P):
            self._evaluate_individual(x)
            self.violation_of_each_constraint[i] = self._get_violation_of_each_constraint(x.g, x.h)
            x.svc = np.sum(self.violation_of_each_constraint, axis=1)[i]
            self.FES += 1

    def before_start(self):
        super().before_start()
        self.constraints = np.argsort(np.mean(self.violation_of_each_constraint, axis=0))  # start with easy ones
        self.Con_n = max(1, np.ceil((self.gn + self.hn) / 2))

    def prepare_to_generate(self):
        super().prepare_to_generate()
        self._update_constraints_num()

    def repair_boundary_constraints(self):
        strategy = np.random.randint(0, 3)
        if strategy == 0:  # for DE
            for v, x in zip(self.T, self.P):
                v = midpoint_target(v, x, self.D, self.MIN, self.MAX)
            
        elif strategy == 1:
            for v, x in zip(self.T, self.P):
                v = min_max_reflection(v, x, self.D, self.MIN, self.MAX)
            
        elif strategy == 2:
            self.T = rand_base_2(self.T, self.D, self.MIN, self.MAX)

    def evaluate_new_population(self):
        svc_det_new = np.zeros((self.NP, max(self.gn + self.hn, 1)))
        for i, u in enumerate(self.O):
            self._evaluate_individual(u)
            svc_det_new[i] = self._get_violation_of_each_constraint(u.g, u.h)
            u.svc = np.sum(svc_det_new[:, self.constraints[self.easy_constraints_index]], axis=1)[i]
            self.FES += 1

    def after_generate(self):
        self._update_probabilities_of_mutations_strategies()
        self._update_memories()

        sorted_index = self._get_sorted_index(self.P)
        self.global_best = self.P[sorted_index[0]]

        worst_individuals = self.P[np.logical_not(self.better_index)]
        self._update_archive(worst_individuals)

        self._update_NP()
        
        super().after_generate()
    
    # helpful methods
    def _update_archive(self, P_new):
        P_with_archive = np.append(self.P_archive, P_new)
        _, index, = np.unique(P_with_archive, return_index=True)
        if len(index) < P_with_archive.shape[0]:
            P_with_archive = P_with_archive[index]
        
        if P_with_archive.shape[0] <= self.NP_archive: 
            self.P_archive = P_with_archive
        else: 
            sorted_index = self._get_sorted_index(P_with_archive)[:self.NP_archive]
            self.P_archive = P_with_archive[sorted_index]  

    def _get_violation_of_each_constraint(self, g, h):
        violation_of_each_constraint = np.zeros(max(self.gn + self.hn, 1))
        violation_of_each_constraint[:self.gn] = np.maximum(0, g)
        violation_of_each_constraint[self.gn:] = np.maximum(np.abs(h) - self.equality_tolerance, 0)

        return violation_of_each_constraint

    def _get_random_indexes(self, NP2):
        r1 = np.zeros(self.NP).astype(int)
        r2 = np.zeros(self.NP).astype(int)
        r3 = np.zeros(self.NP).astype(int)
        
        for i in range(self.NP):
            new_arr = np.delete(np.arange(self.NP), i)
            r1[i] = np.random.choice(new_arr)

            new_arr = np.delete(np.arange(self.NP), (i, r1[i]))
            r2[i] = np.random.choice(new_arr)

            new_arr = np.delete(np.arange(NP2), (i, r1[i], r2[i]))
            r3[i] = np.random.choice(new_arr)
        
        return r1, r2, r3

    def _is_better(self, x1, x2):
        return ((x1.svc == 0) & (x2.svc == 0) & (x1.objective < x2.objective)) | (x1.svc < x2.svc)    
    
    def _reevaluate_archive(self):
        for x in self.P_archive:
            a = self._get_violation_of_each_constraint(x.g, x.h)
            x.svc = np.sum(a)

    def _update_constraints_num(self):
        if self.Con_n <= self.gn + self.hn:
            self.t += 1
        
        if self.t % self.W == 0 and self.Con_n < self.gn + self.hn:
            self.Con_n = min(self.gn + self.hn, self.Con_n + np.ceil((self.gn + self.hn) / 2))
            self._reevaluate_archive()
        
        self.easy_constraints_index = self.constraints[0:int(self.Con_n)]

        violation_of_each_constraint = np.zeros((self.NP, max(self.gn + self.hn, 1)))
        
        for j in range(self.NP):
            violation_of_each_constraint[j] = self._get_violation_of_each_constraint(self.P[j].g, self.P[j].h)
            self.P[j].svc = np.sum(violation_of_each_constraint[:, self.constraints[self.easy_constraints_index]], axis=1)[j]

    def _update_memories(self):
        S_CR = np.array([X.CR for X in self.P[self.better_index]])
        S_F = np.array([X.F for X in self.P[self.better_index]])

        if np.size(S_CR) > 0:
            weights = self.diff[self.better_index] / np.sum(self.diff[self.better_index])
            self.M_F[self.hist_pos] = np.dot(weights, S_F ** 2) / np.dot(weights, S_F)

            if np.max(S_CR) == 0 or self.M_CR[self.hist_pos] == -1:
                self.M_CR[self.hist_pos] = -1
            else:
                self.M_CR[self.hist_pos] = np.dot(weights, S_CR ** 2) / np.dot(weights, S_CR)
            
            self.hist_pos = (self.hist_pos + 1) % self.H
    
    def _generate_F_and_CR(self):
        r = np.floor(self.H * np.random.rand(self.NP)).astype(int)
        F_rand = self.M_F[r]
        CR_rand = self.M_CR[r]
        
        CR = np.random.normal(CR_rand, 0.1)
        CR = np.nanmin([CR, np.ones(CR.shape)], axis=0)
        CR = np.nanmax([CR, np.zeros(CR.shape)], axis=0)
        
        F = F_rand + 0.1 * np.tan(np.pi * (np.random.rand(self.NP) - 0.5))
        pos = np.where(F <= 0)[0]
        while len(pos) > 0:
            F[pos] = F_rand[pos] + 0.1 * np.tan(np.pi * (np.random.rand(len(pos)) - 0.5))
            pos = np.where(F <= 0)[0]
        F = np.nanmin([F, np.ones(F.shape)], axis=0)

        return F, CR

    def _update_probabilities_of_mutations_strategies(self):
        diversity_strategies = np.zeros(self._mutation_strategies_num)
        if self.FES <= self.limit_all:
            for i in range(self.NP):
                diversity_strategies[self.P[i].mutation_num] += self.diff2[i]

        for i in range(self._mutation_strategies_num):
            diversity_strategies[i] = np.nanmax([0, np.sum(diversity_strategies[i]) / self.num_individuals_strategies[i]])

        if np.all(diversity_strategies):
            for i in range(self._mutation_strategies_num):
                diversity_rate = diversity_strategies[i] / np.sum(diversity_strategies)
                self._probability_of_mutation_strategies[i] = np.nanmax([0.1, np.nanmin([0.9, diversity_rate])])
        else:
            self._probability_of_mutation_strategies = np.ones(self._mutation_strategies_num) / self._mutation_strategies_num

    def _check_best_feasibility(self):
        return super()._check_best_feasibility() and self.Con_n == max(self.gn + self.hn, 1)