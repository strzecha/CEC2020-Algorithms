import numpy as np
import copy
from implementations.evolutionary_algorithm import Individual, EvolutionaryAlgorithm

class EnMODE(EvolutionaryAlgorithm):
    def __init__(self, p=0.1, nop=2, prob_ls=0.1, prob_1=1, prob_2=2, archive_rate=2.6):
        super().__init__()
        self.p = 0.1 # p best solutions
        self.nop = nop
        self.p_ls = prob_ls
        self.p_1 = prob_1
        self.p_2 = prob_2
        self.archive_rate = archive_rate

    def initialize_parameters(self, fun, dimensionality, budget_FES, MAX, MIN):
        super().initialize_parameters(fun, dimensionality, budget_FES, MAX, MIN)
        self.NP = 12 * self.D * self.D
        self.archive_size = int(self.archive_rate * self.NP)
        self.archive = list()

    def initialize_population(self):
        self.P = [EnMODEIndividual(np.random.rand(self.D), 0.5, 0.5) for _ in range(self.NP)]

    def evaluate_population(self):
        for i in range(self.NP):
            self.evaluate_individual(self.P[i])

        self.FES += self.NP

    def before_start(self):
        # prepare DE operators
        op_1 = Operator(current_to_pbest_archive, self.NP // 2, self.D)
        #op_1.P = copy.deepcopy(self.P[:self.NP // 2])
        #op_1.x_best = get_pbest(op_1.P)[0]
        op_2 = Operator(rand_to_pbest, self.NP // 2, self.D)
        #op_2.P = copy.deepcopy(self.P[self.NP // 2:])
        #op_2.x_best = get_pbest(op_2.P)[0]

        self.ops = [op_1, op_2]

    def prepare_to_generate_population(self):
        self.pbest = get_pbest(self.P)
        self.new_P = list()

        self.shuffled_P = copy.deepcopy(self.P)

        np.random.shuffle(self.shuffled_P)

        index = 0
        for i in range(self.nop):
            self.ops[i].P = self.shuffled_P[index:index+self.ops[i].NP]
            index = self.ops[i].NP
            self.ops[i].x_best = get_pbest(self.ops[i].P)[0]

    def mutation(self):
        for i in range(len(self.ops)):
            if self.ops[i].NP > 0:
                self.ops[i].mutation(self.P, self.archive, self.pbest)

    def crossover(self):
        for i in range(len(self.ops)):
            if self.ops[i].NP > 0:
                self.ops[i].crossover()
                self.new_P.extend(self.ops[i].P)

        self.P = copy.deepcopy(self.new_P)
    
    def after_generate(self):
        for i in range(len(self.ops)):
            if self.ops[i].NP > 0:
                self.ops[i].calculate_diversity()

        for i in range(len(self.ops)):
                if self.ops[i].NP > 0:
                    self.ops[i].x_best = get_pbest(self.ops[i].P)[0]

        # 9
        for i in range(len(self.ops)):
            if self.ops[i].NP > 0:
                self.ops[i].calculate_diversity_rate(self.ops)
                self.ops[i].calculate_quality_rate(self.ops)
                self.ops[i].calculate_improvment_rate_value()
        
        for i in range(len(self.ops)):
            if self.ops[i].NP > 0:
                self.ops[i].calculate_new_size_of_population(np.delete(self.ops, i), self.NP)

        # 10, 11
        self.archive = update_archive(self.archive, self.new_P, self.archive_size)

        pbest = get_pbest(self.P)
        self.FESs.append(self.FES)
        self.bests_values.append(pbest[0].objective)

        if self.FES >= self.MAX_FES:
            self.stop = True
        
class EnMODEIndividual(Individual):
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
                self.P.append(EnMODEIndividual(np.random.rand(self.dim), 0.5, 0.5))

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

p = 0.1
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

def current_to_pbest_archive(x_i, CR_i, F_i, P, archive, pbest):
    P_A = np.append(P, archive)
    x_pbest = np.random.choice(pbest)
    x_r1 = x_r2 = None
    
    while x_r1 == x_r2:
        x_r1, x_r2 = np.random.choice(P_A, 2)

    return x_i + (x_pbest - x_i) * F_i + (x_r1 - x_r2) * F_i

def rand_to_pbest(x_i, CR_i, F_i, P, archive, pbest):
    P_A = np.append(P, archive)
    x_pbest = np.random.choice(pbest)
    x_r1 = x_r2 = None
    
    while x_r1 == x_r2:
        x_r1, x_r2 = np.random.choice(P, 2)

    return x_i + (x_pbest - x_i + x_r1 - x_r2) * F_i
