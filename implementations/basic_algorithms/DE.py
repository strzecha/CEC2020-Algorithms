from abc import abstractmethod

from implementations.basic_algorithms.evolutionary_algorithm_with_constraints import (EvolutionaryAlgorithmWithConstraints,
                                                                      IndividualWithConstrains)

class DEIndividual(IndividualWithConstrains):
    def __init__(self, x=None, objective=None, svc=None, g=None, h=None, F=None, CR=None):
        super().__init__(x, objective, svc, g, h)
        self.F = F
        self.CR = CR


class DE(EvolutionaryAlgorithmWithConstraints):
    def __init__(self):
        self.individual_generic = DEIndividual

    # main methods
    @abstractmethod
    def mutation(self):
        pass 

    @abstractmethod
    def repair_boundary_constraints(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    def optimize(self, fun, dimensionality, budget_FES, MAX, MIN):
        self.initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.initialize_population()
        self.evaluate_initial_population()
        self.before_start()

        while not self.stop:
            self.prepare_to_generate()
            self.mutation()
            self.repair_boundary_constraints()
            self.crossover()
            self.evaluate_new_population()
            self.selection()
            self.after_generate()

        return self.get_final_results()
