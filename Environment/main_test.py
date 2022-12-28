from implementations.IMODE_2 import IMODE
from implementations.COLSHADE import COLSHADE
from implementations.EnMODE import EnMODE
from implementations.AGSK import AGSK
from implementations.esCMAgES import CMAES
from test_functions.test_functions import fun1D, fun2D, fun3D, hypersphere2D, easom2D, hypersphere5D, hypersphere10D

CMAES(50, fun2D, 2)


#IMODE_main([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fun1D)
IMODE(1, 1000, fun1D)
IMODE(2, 4000, fun2D)
IMODE(3, 9000, fun3D)
#print(fun1D([1.02387573]))
#COLSHADE(1, 1000, fun1D)
#COLSHADE(2, 1000, hypersphere2D)
#COLSHADE(5, 2500, hypersphere5D)
#COLSHADE(10, 10000, hypersphere10D)
"""
AGSK(1, 1000, fun1D)
AGSK(2, 4000, fun2D)
AGSK(3, 9000, fun3D)
AGSK(5, 10000, hypersphere5D)
AGSK(2, 5000, easom2D)
AGSK(10, 10000, hypersphere10D)
"""