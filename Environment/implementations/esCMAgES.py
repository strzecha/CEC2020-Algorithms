import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
from evolutionary_algorithm import Individual

class CMAESIndividual:
    def __init__(self, y, z, d):
        self.y = y
        self.z = z
        self.d = d
        self.objective = 0

    def evaluate(self, fun):
        self.objective = fun(self.y)

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

def esCMAgES(MAXFES, fun, dim, MAX, MIN):
    # 1
    lambda_0 = int(4 + np.floor(3 * np.log(dim)))
    mi = int(np.floor(lambda_0 / 3))
    sigma_0 = 1
    P_sigma = np.zeros([1, dim])
    P_c = np.zeros([1, dim])
    C = np.identity(dim)
    S_1 = np.array([])
    FES = 0
    c_c = 4 / (dim + 4)
    P_sigma
    #teta_p = 0.2
    teta_p = -1
    teta_r = 0
    gamma_min = 3
    sigma_max = 100
    T = 500

    w = np.log([mi + 0.5 for i in range(mi)]).reshape(mi, 1) - np.log([i + 1 for i in range(mi)]).reshape(mi, 1)
    w = w / np.sum(w)

    mi_eff = 1 / np.sum(w ** 2)
    c_sigma = (mi_eff + 2) / (dim + mi_eff + 3)

    c_mi = ((dim + 2) / 3) * ((2 / (mi_eff * (dim + np.sqrt(2)) ** 2)) + (1 - 1 / mi_eff) * min(1, (2 * mi_eff - 1) / ((dim + 2) ** 2 + mi_eff)))

    S_0 = np.array([]) # archive ??

    objs = list()
    FESs = list()

    # 2
    while FES <= MAXFES:
        # 3 - 9
        y = MIN + (MAX - MIN) * np.random.rand(lambda_0, dim)
        # 10
        #e_0 = np.sum([v(y[i]) / np.floor(teta_t * lambda_0) for i in range(np.floor(teta_t * lambda_0))]) 

        # 11
        #gamma = max(gamma_min, (-5 - np.log(e_0)) / np.log(0.05))

        # 12
        FES += lambda_0

        # 13
        x = np.zeros([1, dim])
        for i in range(mi):
            x += w[i] * y[i]

        # 14
        y_best = [y[0]]
        objs.append(fun(y_best))
        FESs.append(0)

        # 15
        g = 0

        # 16
        restart = 1

        # 17
        while FES <= MAXFES and restart == 1:
            # 18
            Y = np.array([])
            D = np.array([])
            Z = np.array([])
            MM = np.diag(np.diag(C) ** 0.5)
            for i in range(lambda_0):
                # 19
                z = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
                # 20
                d = MM @ z
                # 21
                y = x + sigma_0 * d
                # 22
                FES += 1
                # 23
                y_d = box_constraints_repair(y, MIN, MAX) # TODO
                # 24
                if g % dim == 0 and np.random.rand() < teta_p:
                    # 25
                    h = 1
                    # 26
                    while h <= teta_r: # ????
                        # 27
                        y_d = gradient_repair(y)
                        # 28
                        FES += 1
                # 29, 30 end
                # 31
                if not np.array_equal(y, y_d):
                    # 32
                    y = y_d
                    # 33
                    d = (y - x) / sigma_0
                    # 34
                    z = 1 / np.diag(C) * d

                Y = np.append(Y, CMAESIndividual(y, z, d))
            # 35, 36 end
            # 37, 38, 39
            y_best, Y = find_best(Y, fun)
            print(y_best)
            objs.append(y_best.objective)
            FESs.append(FES)
            # 40
            x_impr = np.zeros([1, dim])
            for i in range(mi):
                x_impr += Y[i].d * w[i]
            x = x + sigma_0 * x_impr
            # 41 - mi_eff to w algorytmie mi_w
            P_impr = np.zeros([1, dim])
            for i in range(mi):
                P_impr += Y[i].z * w[i]
            P_sigma = (1 - c_sigma) * P_sigma + np.sqrt(mi_eff * c_sigma * (2 - c_sigma)) * P_impr
            # 42
            h_sigma = (np.linalg.norm(P_sigma) ** 2 / (dim * (1 - (1-c_sigma) ** (2 * FES / lambda_0)))) < (2 + 4 / (dim + 1))
            h_sigma = int(h_sigma)
            # 43
            P_c = (1 - c_c) * P_c + h_sigma * np.sqrt(mi_eff * c_c * (2 - c_c)) * x_impr
            # 44
            C_impr = np.zeros([dim, dim])
            for i in range(mi):
                C_impr = C_impr + w[i] * Y[i].z * Y[i].z.reshape(dim, 1)


            C = (1 - c_mi * (1 - 1 / mi_eff) * C_impr) * C + 1 / mi_eff * P_c * P_c.T
            # 45
            sigma_0 = min(sigma_0 * np.exp(c_sigma / 2 * (np.linalg.norm(P_sigma) ** 2 / dim - 1)), sigma_max)
            # 46
            g += 1
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
                lambda_0 = 1.5 * lambda_0
                # 54
                restart = 0
            """
    plt.plot(FESs, objs)
    plt.show()

def find_best(Y, fun):
    for individual in Y:
        individual.evaluate(fun)

    Y = sorted(Y, key=lambda x: x.objective)

    return Y[0], Y

def gradient_repair():
    pass # TODO

def box_constraints_repair(value, MIN, MAX):
    for i in range(len(value)):
        if value[0][i] < MIN:
            value[0][i] = MIN
        elif value[0][i] > MAX:
            value[0][i] = MAX
    return value

        