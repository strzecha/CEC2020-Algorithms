from abc import ABC, abstractmethod
import numpy as np

class EvolutionaryAlgorithm(ABC):
    def __init__(self, F, CR):
        self.CR = CR
        self.F = F

        self.bests_values = list() # visualization

    def prepare_to_optimize(self):
        self.FESs = list()
        self.bests_values = list() # visualization

    @abstractmethod
    def optimize(self, fun, dimensionality, budget_FES, MAX, MIN):
        pass

    @abstractmethod
    def mutation(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    def initialize_population(self, size, individual, dimensionality):
        return [individual(np.random.rand(dimensionality), 0.5, 0.5) for _ in range(size)]

    def best_of(self, population):
        # looking for the lowest value of objective
        population = sorted(population, key=lambda ind: ind.objective)
        best = population[0]
        return best

    def evaluate(self, individual, function):
        individual.objective = function(individual.x)
        return individual
