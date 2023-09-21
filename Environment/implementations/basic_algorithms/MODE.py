import numpy as np

from implementations.basic_algorithms.DE import DE, DEIndividual


class MODEIndividual(DEIndividual):
    def __init__(self, x=None, objective=None, svc=None, g=None, h=None, F=None, CR=None,
                 mutation_num=None):
        super().__init__(x, objective, svc, g, h, F, CR)
        self.mutation_num = mutation_num

class MODE(DE):
    def __init__(self):
        self.individual_generic = MODEIndividual

    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.p_best_rate = 0.1
        self.P_archive = np.array([])
        self._mutation_strategies = np.array([])

    def before_start(self):
        super().before_start()
        self._mutation_strategies_num = np.size(self._mutation_strategies)
        self._probability_of_mutation_strategies = np.ones(self._mutation_strategies_num) / self._mutation_strategies_num
        
    def prepare_to_generate(self):
        self.num_individuals_strategies = np.zeros(self._mutation_strategies_num)    

    def mutation(self):
        self._assign_mutations_strategies()
        self._generate_parameters()
        
        P_with_archive = np.concatenate((self.P, self.P_archive))
        r1, r2, r3 = self._get_random_indexes(P_with_archive.shape[0])
        p_best_index = self._get_p_best_indexes()
        pbest = self.P[p_best_index]

        self.T = np.array([self.individual_generic(np.zeros(self.D)) for i in range(self.NP)])
        
        for i in range(self.NP):
            mutation_strategy = self._mutation_strategies[self.P[i].mutation_num]
            self.T[i] = mutation_strategy(self.P[i], self.P, P_with_archive, [r1[i], r2[i], r3[i]], pbest[i])

    # helpful methods
    def _count_lehmer_mean(self, S, w):
        return np.sum(w * (S ** 2)) / np.sum(w * S)

    def _assign_mutations_strategies(self):
        for i in range(self.NP):
            r = np.random.rand()
            for j in range(self._mutation_strategies_num):
                if r <= self._probability_of_mutation_strategies[j] + np.sum(self._probability_of_mutation_strategies[:j]):
                    self.P[i].mutation_num = j
                    self.num_individuals_strategies[j] += 1
                    break

    def _generate_parameters(self):
        pass

    def _get_random_indexes(self):
        pass

    def _get_p_best_indexes(self):
        pass
