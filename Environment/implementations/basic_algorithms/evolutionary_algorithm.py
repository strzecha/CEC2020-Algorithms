from abc import ABC, abstractmethod
import numpy as np


class Individual(ABC):
    def __init__(self, x=None, objective=None):
        self.x = x
        self.objective = objective

    def __repr__(self):
        return f"{self.x} = {self.objective}"
    
    def __lt__(self, individual):
        return list(self.x) < list(individual.x)
    
    def __eq__(self, individual):
        return np.array_equal(self.x, individual.x)
    
    def __rmul__(self, num):
        return Individual(self.x * num)
    
    def __add__(self, other):
        return Individual(self.x + other.x)
    
    def __sub__(self, other):
        return Individual(self.x - other.x)
    

class EvolutionaryAlgorithm(ABC):
    def __init__(self):
        self.individual_generic = Individual
        self.name = "Evolutionary Algorithm"

    # main methods
    def initialize_parameters(self, fun, dimensionality, budget_FES):
        self.new_bests = np.empty(2)

        self.fun = fun
        self.D = dimensionality

        self.FES = 0
        self.FES_MAX = budget_FES

        self.FES_reached = budget_FES
        self.optimum = fun.global_minimum

        self.global_best = None

    def initialize_population(self):
        self.P = np.array([self.individual_generic(self.MIN + (self.MAX - self.MIN) * np.random.rand(self.D)) 
                      for i in range(self.NP)])
       
    def evaluate_initial_population(self):
        for i in range(self.NP):
            self._evaluate_individual(self.P[i])
            self.FES += 1

    def before_start(self):
        self.stop = False

    @abstractmethod
    def prepare_to_generate(self):
        pass

    def genetics_operations(self):
        pass

    @abstractmethod
    def selection(self):
        pass
    
    def after_generate(self):
        self._update_bests()
        
        if self._check_optimum_reached():
            self.FES_reached = self.FES
            self.stop = True
            self.global_best.objective = self.optimum

        if self.FES >= self.FES_MAX:
            self.stop = True
    
    def get_final_results(self):
        return self.global_best, self.FES_reached, self.new_bests

    def optimize(self, fun, dimensionality, budget_FES, MAX, MIN):
        self.initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.initialize_population()
        self.evaluate_initial_population()
        self.before_start()

        while not self.stop:
            self.prepare_to_generate()
            self.genetics_operations()
            self.selection()
            self.after_generate()

        return self.get_final_results()
    
    # helpful methods
    def _update_bests(self):
        self.new_bests = np.vstack([self.new_bests, np.array([self.FES, np.abs(self.global_best.objective - self.optimum)])])

    def _evaluate_individual(self, individual):
        individual.objective = self.fun(individual.x.reshape(1, self.D))

    def _get_sorted_index(self, population, svcs=None):
        # sort by objective
        objectives = np.array([x.objective for x in population])
        return np.argsort(objectives)

    def _check_optimum_reached(self):
        if self.optimum is not None:
            return np.abs(self.global_best.objective - self.optimum) <= self.objective_tolerance
        return False
