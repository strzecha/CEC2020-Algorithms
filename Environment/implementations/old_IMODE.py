import numpy as np
import copy
import matplotlib.pyplot as plt
from implementations.evolutionary_algorithm import Individual, EvolutionaryAlgorithm

class IMODEIndividual(Individual):
    def __init__(self, x, CR=0.5, F=0.5):
        super().__init__(x)
        self.CR = CR
        self.F = F

class Operator:
    def __init__(self, strategy, NP_op, dim):
        self.strategy = strategy
        self.NP = NP_op
        self.P = None
        self.D = None
        self.DR = None
        self.QR = None
        self.IRV = None
        self.x_best = None
        self.dim = dim

    def do_crossover(self, x_i, v_i, CR_i):
        j_rand = np.random.randint(0, np.size(x_i))

        for j in range(np.size(x_i)):
            u_i = copy.deepcopy(x_i)
            if np.random.rand() <= 0.3:
                if np.random.rand() <= CR_i or j == j_rand:
                    u_i.x[j] = v_i.x[j]
            
            else:
                u_i.x[j] = x_i.x[j]

        return u_i

    def regenerate(self):
        if self.NP > len(self.P):
            for i in range(self.NP - len(self.P)):
                self.P.append(IMODEIndividual(np.random.rand(self.dim), 0.5, 0.5))

        else:
            bests = sorted(self.P, key=lambda x: x.objective)
            self.P = bests[:self.NP]

    def mutation(self, P, archive, pbest):
        self.O = list()
        self.regenerate()

        for i in range(self.NP):
            v_i = self.strategy(self.P[i], self.P[i].CR, self.P[i].F, P, archive, pbest)
            self.O.append(v_i)

    def crossover(self):
        new_P = list()

        for i in range(self.NP):
            v_i = self.O[i]
            u_i = self.do_crossover(self.P[i], v_i, self.P[i].CR)

            new_P.append(u_i)

        self.P = copy.deepcopy(new_P)

    def generate(self, P, archive, pbest):
        new_P = list()
        self.regenerate()
        for i in range(self.NP):
            v_i = self.strategy(self.P[i], self.P[i].CR, self.P[i].F, P, archive, pbest)
            u_i = self.do_crossover(self.P[i], v_i, self.P[i].CR)

            new_P.append(u_i)

        self.P = copy.deepcopy(new_P)
    
    def calculate_diversity(self):
        self.D = 1 / self.NP * np.sum([np.linalg.norm(self.P[i].x - self.x_best.x) for i in range(self.NP)])

    def calculate_diversity_rate(self, ops):
        self.DR =  self.D / np.sum([op.D for op in ops])

    def calculate_quality_rate(self, ops):
        self.QR = self.x_best.objective / np.sum([op.x_best.objective for op in ops])

    def calculate_improvment_rate_value(self):
        self.IRV = (1 - self.QR) + self.DR

    def calculate_new_size_of_population(self, other_ops, NP):
        self.NP = int(max(0.1, min(0.9, self.IRV / np.sum([op.IRV for op in other_ops]))) * NP)


class IMODE(EvolutionaryAlgorithm):
    def __init__(self):
        self.nop = 3
        self.Prob_ls = 0.1
        self.prob_1 = 1
        self.prob_2 = 1
        self.archive_rate = 2.6

    def crossover(self):
        return super().crossover()

    def mutation(self):
        return super().mutation()

    def selection(self):
        return super().selection()
        
    def optimize(self, fun, dimensionality, budget_FES, MAX, MIN):
        self.prepare_to_optimize()

        NP = 12 * dimensionality * dimensionality
        archive_size = int(self.archive_rate * NP)
        archive = list()

        G = 1
        FES = 0
        MAX_FES = budget_FES

        # 2
        #P = [IMODEIndividual(np.random.rand(dim), np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)) for _ in range(NP)]
        #P = [IMODEIndividual(np.random.rand(dimensionality), 0.5, 0.5) for _ in range(NP)]
        P = self.initialize_population(NP, IMODEIndividual, dimensionality)

        # 3
        for i in range(NP):
                P[i].objective = evaluate(P[i], fun)
        FES += NP

        # 4
        op_1 = Operator(current_to_pbest_archive, NP // 3, dimensionality)
        op_1.P = copy.deepcopy(P[:NP // 3])
        op_1.x_best = get_pbest(op_1.P)[0]
        op_2 = Operator(current_to_pbest_without_archive, NP // 3, dimensionality)
        op_2.P = copy.deepcopy(P[NP // 3:NP // 3 * 2])
        op_2.x_best = get_pbest(op_2.P)[0]
        op_3 = Operator(weighted_to_to_pbest, NP // 3, dimensionality)
        op_3.P = copy.deepcopy(P[NP // 3 * 2:])
        op_3.x_best = get_pbest(op_3.P)[0]

        ops = [op_1, op_2, op_3]


        objs = list()
        # 5
        while FES < MAX_FES:
            # 6
            G += 1

            # 7, 8
            pbest = get_pbest(P)
            new_P = list()

            for i in range(len(ops)):
                if ops[i].NP > 0:
                    ops[i].mutation(P, archive, pbest)
                    ops[i].crossover()
                    new_P.extend(ops[i].P)
                    ops[i].calculate_diversity()

            for i in range(len(ops)):
                if ops[i].NP > 0:
                    ops[i].x_best = get_pbest(ops[i].P)[0]

            # 9
            for i in range(len(ops)):
                if ops[i].NP > 0:
                    ops[i].calculate_diversity_rate(ops)
                    ops[i].calculate_quality_rate(ops)
                    ops[i].calculate_improvment_rate_value()
            
            for i in range(len(ops)):
                if ops[i].NP > 0:
                    ops[i].calculate_new_size_of_population(np.delete(ops, i), NP)

            # 10, 11
            archive = update_archive(archive, new_P, archive_size)
            P = copy.deepcopy(new_P)
            new_P = list()

            pbest = get_pbest(P)
            for i in range(len(ops)):
                if ops[i].NP > 0:
                    ops[i].mutation(P, archive, pbest)
                    ops[i].crossover()
                    #ops[i].generate(P, archive, pbest)
                    new_P.extend(ops[i].P)
                    for j in range(ops[i].NP):
                        ops[i].P[j].objective = evaluate(ops[i].P[j], fun)

            for i in range(len(ops)):
                if ops[i].NP > 0:
                    ops[i].x_best = get_pbest(ops[i].P)[0]

            FES += NP

            # 12 
            # ???

            # 13
            if FES >= 0.85 * MAX_FES and FES < MAX_FES:
                # 14
                SQP()
                # 15
                # ???

            pbest = get_pbest(P)
            self.FESs.append(FES)
            self.bests_values.append(pbest[0].objective)

        return pbest[0].x, pbest[0].objective

def IMODE_fun(dim, MAX_FES, fun):
    # 1
    nop = 3
    Prob_ls = 0.1
    prob_1 = 1
    prob_2 = 1
    NP = 12 * dim * dim
    G = 1
    FES = 0
    archive_rate = 2.6
    archive_size = int(archive_rate * NP)
    archive = list()

    # 2
    #P = [IMODEIndividual(np.random.rand(dim), np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)) for _ in range(NP)]
    P = [IMODEIndividual(np.random.rand(dim), 0.5, 0.5) for _ in range(NP)]

    # 3
    for i in range(NP):
            P[i].objective = evaluate(P[i], fun)
    FES += NP

    # 4
    op_1 = Operator(current_to_pbest_archive, NP // 3, dim)
    op_1.P = copy.deepcopy(P[:NP // 3])
    op_1.x_best = get_pbest(op_1.P)[0]
    op_2 = Operator(current_to_pbest_without_archive, NP // 3, dim)
    op_2.P = copy.deepcopy(P[NP // 3:NP // 3 * 2])
    op_2.x_best = get_pbest(op_2.P)[0]
    op_3 = Operator(weighted_to_to_pbest, NP // 3, dim)
    op_3.P = copy.deepcopy(P[NP // 3 * 2:])
    op_3.x_best = get_pbest(op_3.P)[0]

    ops = [op_1, op_2, op_3]


    objs = list()
    # 5
    while FES < MAX_FES:
        # 6
        G += 1

        # 7, 8
        pbest = get_pbest(P)
        new_P = list()

        for i in range(len(ops)):
            if ops[i].NP > 0:
                ops[i].generate(P, archive, pbest)
                new_P.extend(ops[i].P)
                ops[i].calculate_diversity()

        for i in range(len(ops)):
            if ops[i].NP > 0:
                ops[i].x_best = get_pbest(ops[i].P)[0]

        # 9
        for i in range(len(ops)):
            if ops[i].NP > 0:
                ops[i].calculate_diversity_rate(ops)
                ops[i].calculate_quality_rate(ops)
                ops[i].calculate_improvment_rate_value()
        
        for i in range(len(ops)):
            if ops[i].NP > 0:
                ops[i].calculate_new_size_of_population(np.delete(ops, i), NP)

        # 10, 11
        archive = update_archive(archive, new_P, archive_size)
        P = copy.deepcopy(new_P)
        new_P = list()

        pbest = get_pbest(P)
        for i in range(len(ops)):
            if ops[i].NP > 0:
                ops[i].generate(P, archive, pbest)
                new_P.extend(ops[i].P)
                for j in range(ops[i].NP):
                    ops[i].P[j].objective = evaluate(ops[i].P[j], fun)

        for i in range(len(ops)):
            if ops[i].NP > 0:
                ops[i].x_best = get_pbest(ops[i].P)[0]

        FES += NP

        # 12 
        # ???

        # 13
        if FES >= 0.85 * MAX_FES and FES < MAX_FES:
            # 14
            SQP()
            # 15
            # ???

        print(FES, pbest[0], NP)
        pbest = get_pbest(P)
        objs.append(pbest[0].objective)

def SQP():
    pass

def get_pbest(P):
    best = sorted(P, key=lambda x: x.objective)
    ind = int(np.ceil(p * np.size(P)))
    return best[:ind]

def update_archive(archive, new_P, archive_size):
    archive = np.append(archive, new_P)
    archive = np.unique(archive)

    if np.size(archive) > archive_size:
        archive = np.sort(archive)
        archive = archive[:archive_size]

    #return archive
    return new_P

def evaluate(x, fun):
    return fun(x.x)

def current_to_pbest_archive(x_i, CR_i, F_i, P, archive, pbest):
    P_A = np.append(P, archive)
    x_pbest = np.random.choice(pbest)
    x_r1 = x_r2 = None
    
    while x_r1 == x_r2:
        x_r1, x_r2 = np.random.choice(P_A, 2)

    return x_i + (x_pbest - x_i) * F_i + (x_r1 - x_r2) * F_i

def current_to_pbest_without_archive(x_i, CR_i, F_i, P, archive, pbest):
    x_pbest = np.random.choice(pbest)
    x_r1 = x_r3 = None
    
    while x_r1 == x_r3:
        x_r1, x_r3 = np.random.choice(P, 2)

    return x_i + (x_pbest - x_i) * F_i + (x_r1 - x_r3) * F_i

def weighted_to_to_pbest(x_i, CR_i, F_i, P, archive, pbest):
    x_pbest = np.random.choice(pbest)
    x_r1 = x_r3 = None
    
    while x_r1 == x_r3:
        x_r1, x_r3 = np.random.choice(P, 2)

    return x_r1 * F_i + (x_pbest - x_r3)