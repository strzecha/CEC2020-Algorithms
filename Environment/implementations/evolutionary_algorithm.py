from abc import ABC, abstractmethod
import numpy as np

class EvolutionaryAlgorithm2(ABC):
    def __init__(self):
        # visualization
        self.bests_values = None 
        self.FESs = None 

    def prepare_to_optimize(self):
        self.FESs = list()
        self.bests_values = list()
        self.pbest = None

    def optimize(self, fun, dimenstionality, budget_FES, MAX, MIN):
        self.prepare_to_optimize()
        self.initialize_parameters(fun, dimenstionality, budget_FES, MAX, MIN)
        self.initialize_population()
        self.evaluate_population()
        self.before_start()

        self.stop = False

        while not self.stop:
            self.prepare_to_generate_population()

            self.mutation()
            self.crossover()
            self.evaluate_population()

            self.operation_after_generate()

        return self.pbest[0].x, self.pbest[0].objective

    @abstractmethod
    def before_start(self):
        pass

    @abstractmethod
    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        self.fun = fun
        self.D = dimensionality

        self.t = 0
        self.FES = 0
        self.MAX_FES = budget_FES

    @abstractmethod
    def initialize_population(self):
        pass

    @abstractmethod
    def evaluate_population(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def prepare_to_generate_population():
        pass

    @abstractmethod
    def operation_after_generate():
        pass

    def evaluate_individual(self, individual):
        individual.evaluate(self.fun)


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


class Individual:
    def __init__(self, x):
        self.x = x
        self.objective = 0

    def __add__(self, individual):
        return Individual(self.x + individual.x)

    def __sub__(self, individual):
        return Individual(self.x - individual.x)

    def __mul__(self, num):
        return Individual(self.x * num)

    def __repr__(self):
        return f"{self.x}: obj:{self.objective}"

    def __lt__(self, individual):
        return list(self.x) < list(individual.x)

    def evaluate(self, fun):
        self.objective = fun(self.x)