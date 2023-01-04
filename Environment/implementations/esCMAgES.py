import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
from implementations.evolutionary_algorithm import Individual, EvolutionaryAlgorithm

class CMAESIndividual:
    def __init__(self, y, z, d):
        self.x = y
        self.z = z
        self.d = d
        self.objective = 0

    def evaluate(self, fun):
        self.objective = fun(self.x)

    def __repr__(self):
        return f"{self.y} {self.z} {self.d}: obj:{self.objective}"

    def __add__(self, individual):
        return CMAESIndividual(self.x + individual.x)
    
    def __sub__(self, individual):
        return CMAESIndividual(self.x - individual.x)

    def __mul__(self, num):
        return CMAESIndividual(self.x * num)

    def __lt__(self, individual):
        return list(self.x) < list(individual.x)

class esCMAgES(EvolutionaryAlgorithm):
    def __init__(self, teta_p=-1, teta_r=0, gamma_min=3, sigma_max=100, T=500):
        #teta_p = 0.2
        self.teta_p = teta_p
        self.teta_r = teta_r
        self.gamma_min = gamma_min
        self.sigma_max = sigma_max
        self.T = T

    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        # 1
        self.NP = int(4 + np.floor(3 * np.log(self.D)))
        self.mi = int(np.floor(self.NP / 3))
        self.sigma_0 = 1
        self.P_sigma = np.zeros([1, self.D])
        self.P_c = np.zeros([1, self.D])
        self.C = np.identity(self.D)
        self.S_1 = np.array([])

        self.c_c = 4 / (self.D + 4)

        self.w = np.log([self.mi + 0.5 for i in range(self.mi)]).reshape(self.mi, 1) - np.log([i + 1 for i in range(self.mi)]).reshape(self.mi, 1)
        self.w = self.w / np.sum(self.w)

        self.mi_eff = 1 / np.sum(self.w ** 2)
        self.c_sigma = (self.mi_eff + 2) / (self.D + self.mi_eff + 3)

        self.c_mi = ((self.D + 2) / 3) * ((2 / (self.mi_eff * (self.D + np.sqrt(2)) ** 2)) + (1 - 1 / self.mi_eff) * min(1, (2 * self.mi_eff - 1) / ((self.D + 2) ** 2 + self.mi_eff)))

        S_0 = np.array([]) # archive ??

    def initialize_population(self):
        self.y = self.MIN + (self.MAX - self.MIN) * np.random.rand(self.NP, self.D)
    
    def evaluate_population(self):
        pass

    def before_start(self):
        self.x = np.zeros([1, self.D])
        for i in range(self.mi):
            self.x += self.w[i] * self.y[i]

        self.g = 0
        self.restart = 1

        y_best = [self.y[0]]
        self.bests_values.append(self.fun(y_best))
        self.FESs.append(0)

    def mutation(self):
        self.Y = np.array([])

        self.MM = np.diag(np.diag(self.C) ** 0.5)
        for i in range(self.NP):
            # 19
            z = np.random.multivariate_normal(np.zeros(self.D), np.identity(self.D))
            # 20
            d = self.MM @ z
            # 21
            y = self.x + self.sigma_0 * d
            # 22
            self.FES += 1
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
                d = (y - self.x) / self.sigma_0
                # 34
                z = 1 / np.diag(self.C) * d

            self.Y = np.append(self.Y, CMAESIndividual(y, z, d))

    def crossover(self):
        return super().crossover()

    def selection(self):
        return super().selection()

    def prepare_to_generate_population(self):
        print(self.NP, self.FES)

    def after_generate(self):
        self.find_best()
        
        # 40
        x_impr = np.zeros([1, self.D])
        for i in range(self.mi):
            x_impr += self.Y[i].d * self.w[i]
        self.x = self.x + self.sigma_0 * x_impr
        # 41 - mi_eff to w algorytmie mi_w
        P_impr = np.zeros([1, self.D])
        for i in range(self.mi):
            P_impr += self.Y[i].z * self.w[i]
        self.P_sigma = (1 - self.c_sigma) * self.P_sigma + np.sqrt(self.mi_eff * self.c_sigma * (2 - self.c_sigma)) * P_impr
        # 42
        h_sigma = (np.linalg.norm(self.P_sigma) ** 2 / (self.D * (1 - (1 - self.c_sigma) ** (2 * self.FES / self.NP)))) < (2 + 4 / (self.D + 1))
        h_sigma = int(h_sigma)
        # 43
        self.P_c = (1 - self.c_c) * self.P_c + h_sigma * np.sqrt(self.mi_eff * self.c_c * (2 - self.c_c)) * x_impr
        # 44
        C_impr = np.zeros([self.D, self.D])
        for i in range(self.mi):
            C_impr = C_impr + self.w[i] * self.Y[i].z * self.Y[i].z.reshape(self.D, 1)


        self.C = (1 - self.c_mi * (1 - 1 / self.mi_eff) * C_impr) * self.C + 1 / self.mi_eff * self.P_c * self.P_c.T
        # 45
        self.sigma_0 = min(self.sigma_0 * np.exp(self.c_sigma / 2 * (np.linalg.norm(self.P_sigma) ** 2 / self.D - 1)), self.sigma_max)
        # 46
        self.g += 1
        # 47
        #if g < T:
            # 48
            #e = e_0 * (1 - g/T) ** gamma
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

    def find_best(self):
        for individual in self.Y:
            individual.evaluate(self.fun)

        self.FES += self.NP
        if self.FES >= self.MAX_FES:
            self.stop = True

        self.Y = sorted(self.Y, key=lambda x: x.objective)
        self.global_best = self.Y[0]

        self.bests_values.append(self.global_best.objective)
        self.FESs.append(self.FES)

    def gradient_repair():
        pass # TODO

    def box_constraints_repair(self, value):
        for i in range(len(value)):
            if value[0][i] < self.MIN:
                value[0][i] = self.MIN
            elif value[0][i] > self.MAX:
                value[0][i] = self.MAX
        return value