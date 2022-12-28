from abc import ABC, abstractmethod


class EvolutionaryAlgorithm(ABC):
    def __init__(self, F, CR):
        self.CR = CR
        self.F = F

        self.bests_values = list() # visualization

    def prepare_to_optimize(self):
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

    @abstractmethod
    def initialize_population(self, size):
        pass

    def best_of(self, population):
        # looking for the lowest value of objective
        population = sorted(population, key=lambda ind: ind.objective)
        best = population[0]
        return best

    def evaluate(self, individual, function):
        individual.objective = function(individual.x)
        return individual


class Individual:
    def __init__(self, x):
        self.x = x
        self.objective = 0

    @abstractmethod
    def __add__(self, individual):
        return Individual(self.x + individual.x)

    @abstractmethod
    def __sub__(self, individual):
        return Individual(self.x - individual.x)

    @abstractmethod
    def __mul__(self, num):
        return Individual(self.x * num)