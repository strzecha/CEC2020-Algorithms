from implementations.j2020 import j2020
from implementations.IMODE import IMODE
from implementations.EnMODE import EnMODE
from implementations.COLSHADE import COLSHADE
from implementations.AGSK import AGSK
from test_functions.test_functions import (fun1D, fun2D, fun3D, 
                                            hypersphere2D, easom2D, 
                                            schwefel2D, himmelblau2D, hypersphere5D)
from utils.visualization import draw_graph_evaluations

alg_1 = IMODE()
alg_2 = EnMODE()
alg_3 = j2020(50, 10)
alg_4 = COLSHADE()
alg_5 = AGSK()



print("Fun1D:")
(best_sol_1, objective_val_1) = alg_1.optimize(fun1D, 1, 500, 100, -100)
(best_sol_2, objective_val_2) = alg_2.optimize(fun1D, 1, 500, 100, -100)
(best_sol_3, objective_val_3) = alg_3.optimize(fun1D, 1, 500, 1, 0)
(best_sol_4, objective_val_4) = alg_4.optimize(fun1D, 1, 500, 100, -100)
(best_sol_5, objective_val_5) = alg_5.optimize(fun1D, 1, 500, 100, -100)
print(best_sol_1, objective_val_1)
print(best_sol_2, objective_val_2)
print(best_sol_3, objective_val_3)
print(best_sol_4, objective_val_4)
print(best_sol_5, objective_val_5)

draw_graph_evaluations([(alg_1, "IMODE"), (alg_2, "EnMODE"), (alg_3, "j2020"), (alg_4, "COLSHADE"), (alg_5, "AGSK")])

print(20 * "-")
print("Fun2D")
(best_sol_1, objective_val_1) = alg_1.optimize(fun2D, 2, 500, 100, -100)
(best_sol_2, objective_val_2) = alg_2.optimize(fun2D, 2, 500, 100, -100)
(best_sol_3, objective_val_3) = alg_3.optimize(fun2D, 2, 500, 1, 0)
(best_sol_4, objective_val_4) = alg_4.optimize(fun2D, 2, 500, 100, -100)
(best_sol_5, objective_val_5) = alg_5.optimize(fun2D, 2, 500, 100, -100)
print(best_sol_1, objective_val_1)
print(best_sol_2, objective_val_2)
print(best_sol_3, objective_val_3)
print(best_sol_4, objective_val_4)
print(best_sol_5, objective_val_5)

draw_graph_evaluations([(alg_1, "IMODE"), (alg_2, "EnMODE"), (alg_3, "j2020"), (alg_4, "COLSHADE"), (alg_5, "AGSK")])


print(20 * "-")
print("Fun3D")
(best_sol_1, objective_val_1) = alg_1.optimize(fun3D, 3, 1000, 100, -100)
(best_sol_2, objective_val_2) = alg_2.optimize(fun3D, 3, 1000, 100, -100)
(best_sol_3, objective_val_3) = alg_3.optimize(fun3D, 3, 1000, 1, 0)
(best_sol_4, objective_val_4) = alg_4.optimize(fun3D, 3, 1000, 100, -100)
(best_sol_5, objective_val_5) = alg_5.optimize(fun3D, 3, 1000, 100, -100)
print(best_sol_1, objective_val_1)
print(best_sol_2, objective_val_2)
print(best_sol_3, objective_val_3)
print(best_sol_4, objective_val_4)
print(best_sol_5, objective_val_5)

draw_graph_evaluations([(alg_1, "IMODE"), (alg_2, "EnMODE"), (alg_3, "j2020"), (alg_4, "COLSHADE"), (alg_5, "AGSK")])


print(20 * "-")
print("Hypersphere5D")
(best_sol_1, objective_val_1) = alg_1.optimize(hypersphere5D, 5, 3000, 100, -100)
(best_sol_2, objective_val_2) = alg_2.optimize(hypersphere5D, 5, 3000, 100, -100)
(best_sol_3, objective_val_3) = alg_3.optimize(hypersphere5D, 5, 3000, 1, 0)
(best_sol_4, objective_val_4) = alg_4.optimize(hypersphere5D, 5, 3000, 100, -100)
(best_sol_5, objective_val_5) = alg_5.optimize(hypersphere5D, 5, 3000, 100, -100)
print(best_sol_1, objective_val_1)
print(best_sol_2, objective_val_2)
print(best_sol_3, objective_val_3)
print(best_sol_4, objective_val_4)
print(best_sol_5, objective_val_5)

draw_graph_evaluations([(alg_1, "IMODE"), (alg_2, "EnMODE"), (alg_3, "j2020"), (alg_4, "COLSHADE"), (alg_5, "AGSK")])
