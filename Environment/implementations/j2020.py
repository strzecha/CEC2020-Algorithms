import numpy as np

from implementations.evolutionary_algorithm import EvolutionaryAlgorithm, Individual

class j2020Individual(Individual):
    def __init__(self, x, F, CR):
        super().__init__(x)
        self.F = F
        self.CR = CR

    def __add__(self, individual):
        return j2020Individual(self.x + individual.x, self.F, self.CR)

    def __sub__(self, individual):
        return j2020Individual(self.x - individual.x, self.F, self.CR)

    def __mul__(self, num):
        return j2020Individual(self.x * num, self.F, self.CR)

class j2020Algorithm(EvolutionaryAlgorithm):
    def __init__(self, bNP=None, sNP=None, F=0.5, Flb=0.01, Fls=0.17, Fu=1.1, tau1=0.1, CR=0.9, tau2=0.1, eps=10**(-16), myEqs=25):
        super().__init__(F, CR)
        self.bNP = bNP
        self.sNP = sNP
        self.m = 0
        self.noImprove = 0
        self.Flb = Flb
        self.Fls = Fls
        self.Fu = Fu
        self.eps = eps
        self.myEqs = myEqs
        self.ageLmt = 0
        self.tau1 = tau1
        self.tau2 = tau2

    def optimize(self, function, dimensionality, budget_FES, MAX, MIN):
        self.prepare_to_optimize()
        # initialization
        self.dimensionality = dimensionality

        self.bNP = self.bNP if self.bNP else 7 * self.dimensionality
        self.sNP = self.sNP if self.sNP else self.dimensionality

        MIN_SNP = 3 # various individuals in mutation

        self.sNP = max(MIN_SNP, self.sNP)

        self.m = self.bNP // self.sNP

        self.max_FES = budget_FES
        self.ageLmt = self.max_FES // 10
        self.noImprove = 0

        self.min = MIN
        self.max = MAX

        self.Pb = self.initialize_population(self.bNP, self.F, self.CR)
        self.Ps = self.initialize_population(self.sNP, self.F, self.CR)

        self.FES = 0
        # evaluation
        for i in range(self.bNP):
            self.Pb[i] = self.evaluate(self.Pb[i], function)
        for i in range(self.sNP):
            self.Ps[i] = self.evaluate(self.Ps[i], function)

        # algorithm
        while self.FES < self.max_FES:
            self.check_reinitialize_big()
            self.check_reinitialize_small()

            Ms_size = 1
            if self.FES > self.max_FES / 3:
                Ms_size = 2
            elif self.FES > self.max_FES * 2 / 3:
                Ms_size = 3

            new_Pb = list()
            for x in self.Pb:
                Ms = np.array([self.Ps[i] for i in np.random.randint(0, self.sNP, Ms_size)])
                v = self.mutation_big(x, Ms)
                u = self.crossover(x, v)
                u = self.evaluate(u, function)
                xi = self.selection(x, u) 

                new_Pb.append(xi)

            self.Pb = np.array(new_Pb)

            best = self.best_of(self.Pb)
            self.FESs.append(self.FES)
            self.bests_values.append(function(best.x))

            self.check_if_improve(best, function)


            for k in range(self.m):
                new_Ps = list()

                for x in self.Ps:
                    v = self.mutation_small(x)
                    u = self.crossover(x, v)
                    u = self.evaluate(u, function)
                    xi = self.selection(x, u)
                    new_Ps.append(xi)

                self.Ps = np.array(new_Ps)

            self.FES += self.bNP * 2

        best = self.best_of(self.Pb)
        return best.x, best.objective

    def check_if_improve(self, current_best, function):
        if function(current_best.x) < function(self.best_of(self.Ps).x): 
            self.Ps[self.FES % self.Ps.shape[0]] = current_best
            self.noImprove = 0
        else:
            self.noImprove += 1

    def check_reinitialize_big(self):
        self.Pb = np.array(sorted(self.Pb, key=lambda ind: ind.objective))

        self.Pb = self.check_reinitialize(self.Pb, self.noImprove >= self.ageLmt)

    def check_reinitialize_small(self):
        self.Ps = np.array(sorted(self.Ps, key=lambda ind: ind.objective))
        best = self.Ps[0]

        self.Ps = self.check_reinitialize(self.Ps)
        self.Ps[-1] = best

    def initialize_population(self, size, F, CR):
        population = [j2020Individual(np.random.uniform(self.min, self.max, self.dimensionality), F, CR) for _ in range(size)]
        return np.array(population)

    def check_reinitialize(self, P, statement=True):
        first = 0
        last = int(P.shape[0] * self.myEqs / 100)

        dif = P[first].objective - P[last].objective

        if abs(dif) < self.eps and statement:
            new_P = np.array([j2020Individual(np.random.uniform(self.min, self.max, self.dimensionality), P[i].F, P[i].CR) for i in range(P.shape[0])])
            P = new_P

        return P

    def mutation_big(self, x, Ms):
        return self.mutation(x, self.Pb, self.Flb, Ms)

    def mutation_small(self, x):
        return self.mutation(x, self.Ps, self.Fls)

    def mutation(self, x, P, Fl, Ms=None):
        # jDE mutation
        F = Fl + np.random.random() * self.Fu if np.random.random() < self.tau1 else x.F
        r1 = r2 = r3 = 0
        
        new_P = np.concatenate((P, Ms)) if Ms is not None else P

        while r1 == r2 or r2 == r3 or r1 == r3:
            r1 = np.random.randint(0, P.shape[0])
            r2 = np.random.randint(0, new_P.shape[0])
            r3 = np.random.randint(0, new_P.shape[0])

        v = new_P[r1] + (new_P[r2] - new_P[r3]) * F
        v.F = F

        return v

    def crossover(self, x, v):
        # jDE crossover
        CR = np.random.random() if np.random.random() < self.tau2 else x.CR
        
        u = j2020Individual(v.x, v.F, CR)
        jrand = np.random.randint(0, self.dimensionality - 1) if self.dimensionality > 1 else 0

        for j in range(self.dimensionality):
            if np.random.random() <= CR or j == jrand:
                u.x[j] = v.x[j]
            else:
                u.x[j] = x.x[j]
        return u

    def selection(self, x, u):
        # jDE selection
        if u.objective <= x.objective:
            xi = u # x, F, CR
        else:
            xi = x # x, F, CR
        return xi
         