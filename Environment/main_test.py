from implementations.j2020 import j2020
from implementations.IMODE import IMODE
from implementations.EnMODE import EnMODE
from implementations.COLSHADE import COLSHADE
from implementations.AGSK import AGSK
from test_functions.test_functions import (fun1D, fun2D, fun3D, 
                                            hypersphere2D, easom2D, 
                                            schwefel2D, himmelblau2D, hypersphere5D)
from utils.visualization import draw_graph_evaluations

alg_1 = COLSHADE()



print("Fun1D:")

(best_sol_1, objective_val_1) = alg_1.optimize(fun1D, 1, 1000, [100], [-100])
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, "COLSHADE")])

print("Fun2D:")

(best_sol_1, objective_val_1) = alg_1.optimize(fun2D, 2, 5000, [100], [-100])
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, "COLSHADE")])

print("Hypersphere5D:")

(best_sol_1, objective_val_1) = alg_1.optimize(hypersphere5D, 5, 10000, [100], [-100])
print(best_sol_1, objective_val_1)
draw_graph_evaluations([ (alg_1, "COLSHADE")])