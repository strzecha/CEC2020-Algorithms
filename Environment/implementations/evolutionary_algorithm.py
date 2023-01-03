from abc import ABC, abstractmethod
import numpy as np

class EvolutionaryAlgorithm(ABC):
    def prepare_to_optimize(self):
        self.FESs = list()
        self.bests_values = list()
        self.global_best = None

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
            self.selection()

            self.after_generate()

        return self.global_best.x, self.global_best.objective

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
    def prepare_to_generate_population(self):
        pass

    @abstractmethod
    def after_generate(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    def evaluate_individual(self, individual):
        individual.evaluate(self.fun)

    def LPSR(self, NP_min, NP_init, MAX_FES, FES):
        NP = int(np.round(((NP_min - NP_init) / MAX_FES) * FES + NP_init))
        return NP

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