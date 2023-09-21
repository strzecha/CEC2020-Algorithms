import numpy as np

from implementations.basic_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm, Individual

class IndividualWithConstrains(Individual):
    def __init__(self, x=None, objective=None, svc=None, g=None, h=None):
        super().__init__(x, objective)
        self.svc = svc
        self.g = g
        self.h = h

    def __repr__(self):
        return f"{self.x} = {self.objective}, {self.svc}"

class EvolutionaryAlgorithmWithConstraints(EvolutionaryAlgorithm):
    def __init__(self):
        self.individual_generic = IndividualWithConstrains

    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES)
        self.new_bests = np.empty(3)

        self.MIN = MIN
        self.MAX = MAX

        self.equality_tolerance = 1e-4
        self.objective_tolerance = 1e-8
        self.gn = fun.inequality_constraints_num
        self.hn = fun.equality_constraints_num

        if self.hn + self.gn > 0:
            self.is_feasible = False
            self.FES_feasible = budget_FES
        else:
            self.is_feasible = True
            self.FES_feasible = 0

    def evaluate_initial_population(self):
        for x in self.P:
            self._evaluate_individual(x)
            x.svc = self._get_svc(x.g, x.h)
            self.FES += 1

    def evaluate_new_population(self):
        for u in self.O:
            self._evaluate_individual(u)
            u.svc = self._get_svc(u.g, u.h)
            self.FES += 1
    
    def after_generate(self):
        if self._check_best_feasibility():
            self.FES_feasible = self.FES
            self.is_feasible = True

        super().after_generate()

    def get_final_results(self):
        return self.global_best, self.FES_feasible, self.FES_reached, self.new_bests

    # helpful methods
    def _update_bests(self):
        self.new_bests = np.vstack([self.new_bests, 
                                    np.array([
                                        self.FES, 
                                        np.abs(self.global_best.objective - self.optimum),
                                        self.global_best.svc / max(self.gn + self.hn, 1)
                                        ])])

    def _is_better(self, x1, x2):
        return ((x1.svc == 0) & (x2.svc == 0) & (x1.objective < x2.objective)) | (x1.svc < x2.svc)
    
    def _get_svc(self, g, h):
        g = np.array([np.maximum(G, 0) for G in g])
        h = np.array([np.maximum(np.abs(H) - self.equality_tolerance, 0) for H in h])
        svc = np.sum(g) + np.sum(h)
        
        return svc
    
    def _evaluate_individual(self, individual):
        individual.objective, individual.g, individual.h = self.fun(individual.x.reshape(1, self.D))

    def _get_sorted_index(self, population, svcs=None):
        # sort by svc and objective
        objectives = np.array([x.objective for x in population])
        if svcs is None:
            svcs = np.array([x.svc for x in population])
        return np.lexsort(np.column_stack((svcs, objectives)).T[::-1])
    
    def _check_best_feasibility(self):
        return self.global_best.svc == 0 and not self.is_feasible
        
    def _check_optimum_reached(self):
        return super()._check_optimum_reached() and self.global_best.svc == 0
    