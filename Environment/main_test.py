from implementations.j2020 import j2020
from implementations.IMODE import IMODE
from implementations.EnMODE import EnMODE
from implementations.COLSHADE import COLSHADE
from implementations.AGSK import AGSK, AGSK2
from test_functions.test_functions import (fun1D, fun2D, fun3D, 
                                            hypersphere2D, easom2D, 
                                            schwefel2D, himmelblau2D, hypersphere5D)
from utils.visualization import draw_graph_evaluations

alg_5 = AGSK2()



print("Fun1D:")

#(best_sol_5, objective_val_5) = alg_5.optimize(fun1D, 1, 500, 100, -100)

#print(best_sol_5, objective_val_5)

#draw_graph_evaluations([ (alg_5, "AGSK")])

(best_sol_5, objective_val_5) = alg_5.optimize(fun3D, 3, 1000, 100, -100)

print(best_sol_5, objective_val_5)

draw_graph_evaluations([ (alg_5, "AGSK")])