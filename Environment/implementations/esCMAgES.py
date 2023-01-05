import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
from implementations.evolutionary_algorithm import Individual, EvolutionaryAlgorithm

class esCMAgESIndividual:
    def __init__(self, y):
        self.x = y
        self.objective = 0

    def evaluate(self, fun):
        self.objective = fun(self.x)

    def __repr__(self):
        return f"{self.y} {self.z} {self.d}: obj:{self.objective}"

    def __lt__(self, individual):
        return list(self.x) < list(individual.x)

class esCMAgES(EvolutionaryAlgorithm):
    def __init__(self, teta_p=-1, teta_r=0, gamma_min=3, sigma_max=100, Tg=500):
        #teta_p = 0.2
        self.teta_p = teta_p
        self.teta_r = teta_r
        self.gamma_min = gamma_min
        self.sigma_max = sigma_max
        self.Tg = Tg

    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        # 1
        self.NP = int(4 + np.floor(3 * np.log(self.D)))
        self.mi = int(np.floor(self.NP / 3))
        self.sigma = 1
        self.P_sigma = np.zeros([1, self.D])
        self.P_c = np.zeros([1, self.D])
        self.C = np.identity(self.D)
        self.S_1 = np.array([])

        self.c_c = 4 / (self.D + 4)

        self.weights = np.log([self.mi + 0.5 for i in range(self.mi)]).reshape(self.mi, 1) - np.log([i + 1 for i in range(self.mi)]).reshape(self.mi, 1)
        self.weights = self.weights / np.sum(self.weights)

        self.mi_eff = 1 / np.sum(self.weights ** 2)
        self.c_sigma = (self.mi_eff + 2) / (self.D + self.mi_eff + 3)

        self.c_mi = ((self.D + 2) / 3) * ((2 / (self.mi_eff * (self.D + np.sqrt(2)) ** 2)) + (1 - 1 / self.mi_eff) * min(1, (2 * self.mi_eff - 1) / ((self.D + 2) ** 2 + self.mi_eff)))

        S_0 = np.array([]) # archive ??

    def initialize_population(self):
        self.P = [esCMAgESIndividual(np.random.uniform(self.MIN, self.MAX, self.D)) for i in range(self.NP)]

    def evaluate_initial_population(self):
        for i in range(self.NP):
            self.evaluate_individual(self.P[i])

        self.FES += self.NP

    def evaluate_new_population(self):
        self.evaluate_initial_population()

    def before_start(self):
        self.x = np.zeros([1, self.D])
        for i in range(self.mi):
            self.x += self.weights[i] * self.P[i].x

        self.g = 0
        self.restart = 1

    def mutation(self):
        self.P = np.array([])
        self.T = np.array([])
        self.O = np.array([])

        self.MM = np.diag(np.diag(self.C) ** 0.5)
        for i in range(self.NP):
            # 19
            z = np.random.multivariate_normal(np.zeros(self.D), np.identity(self.D))
            # 20
            d = self.MM @ z
            # 21
            y = self.x + self.sigma * d
            # 23
            y_d = self.box_constraints_repair(y) # TODO
            # 24
            if self.g % self.D == 0 and np.random.rand() < self.teta_p:
                # 25
                h = 1
                # 26
                while h <= self.teta_r: # ????
                    # 27
                    y_d = gradient_repair(y)
                    # 28
                    self.FES += 1
            # 29, 30 end
            # 31
            if not np.array_equal(y, y_d):
                # 32
                y = y_d
                # 33
                d = (y - self.x) / self.sigma
                # 34
                z = 1 / np.diag(self.C) * d

            self.P = np.append(self.P, esCMAgESIndividual(y[0]))
            self.T = np.append(self.T, esCMAgESIndividual(z[0]))
            self.O = np.append(self.O, esCMAgESIndividual(d[0]))

    def crossover(self):
        pass

    def selection(self):
        pass

    def prepare_to_generate_population(self):
        pass

    def after_generate(self):
        self.find_best()
        
        # 40
        x_impr = np.zeros([1, self.D])
        for i in range(self.mi):
            x_impr += self.O[i].x * self.weights[i]
        self.x = self.x + self.sigma * x_impr
        # 41 - mi_eff to w algorytmie mi_w
        P_impr = np.zeros([1, self.D])
        for i in range(self.mi):
            P_impr += self.T[i].x * self.weights[i]
        self.P_sigma = (1 - self.c_sigma) * self.P_sigma + np.sqrt(self.mi_eff * self.c_sigma * (2 - self.c_sigma)) * P_impr
        # 42
        h_sigma = (np.linalg.norm(self.P_sigma) ** 2 / (self.D * (1 - (1 - self.c_sigma) ** (2 * self.FES / self.NP)))) < (2 + 4 / (self.D + 1))
        h_sigma = int(h_sigma)
        # 43
        self.P_c = (1 - self.c_c) * self.P_c + h_sigma * np.sqrt(self.mi_eff * self.c_c * (2 - self.c_c)) * x_impr
        # 44
        C_impr = np.zeros([self.D, self.D])
        for i in range(self.mi):
            C_impr = C_impr + self.weights[i] * self.T[i].x * self.T[i].x.reshape(self.D, 1)


        self.C = (1 - self.c_mi * (1 - 1 / self.mi_eff) * C_impr) * self.C + 1 / self.mi_eff * self.P_c * self.P_c.T
        # 45
        self.sigma = min(self.sigma * np.exp(self.c_sigma / 2 * (np.linalg.norm(self.P_sigma) ** 2 / self.D - 1)), self.sigma_max)
        # 46
        self.g += 1
        # 47
        #if self.g < self.Tg:
            # 48
            #e = e_0 * (1 - self.g/self.Tg) ** gamma
        # 49
        #else:
            # 50
            #e = 0
        # 51 end
        # 52
        """
        if stop:
            # 53
            self.NP = 1.5 * self.NP
            # 54
            restart = 0
        """

        if self.FES >= self.MAX_FES:
            self.stop = True

        self.bests_values.append(self.global_best.objective)
        self.FESs.append(self.FES)

    def find_best(self):
        self.P, self.T, self.O = zip(*sorted(zip(self.P, self.T, self.O), key=lambda x: x[0].objective))
        self.global_best = self.P[0]

    def gradient_repair():
        pass # TODO

    def box_constraints_repair(self, value):
        for i in range(len(value)):
            value[0][i] = np.min([np.max([value[0][i], self.MIN[i]]), self.MAX[i]])
        return value