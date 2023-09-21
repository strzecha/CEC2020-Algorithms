import ctypes
import numpy as np

from test_functions.test_function import TestFunction
from utils.cec20_parameters import get_cec2020_nonconstrained_optimum
from utils.data_reader import get_properties

properties = get_properties()
SYSTEM = properties.get("SYSTEM").data

if SYSTEM == "windows":
    func = ctypes.cdll.LoadLibrary('test_functions/cec2020.dll')
if SYSTEM == "linux":
    func = ctypes.cdll.LoadLibrary('test_functions/cec2020.so')

func.cec20_test_func.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int)
func.cec20_test_func.restype = None


def cec20_func(x, func_num, dim=None):
    if dim is None:
        dim = np.size(x[0])
    res = np.zeros(x.shape[0], dtype=np.float64)

    func.cec20_test_func(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                            ctypes.c_int(dim), ctypes.c_int(np.size(res)), ctypes.c_int(func_num))
    return res

def cec20_func_factory(dim):
    funcs = []
    optimum = get_cec2020_nonconstrained_optimum()
    for i in range(10):
        opt = optimum[i]
        fun = TestFunction(f"F{i+1}", lambda x, idx=i: (cec20_func(x.reshape((1, dim)), idx+1)[0], np.zeros((x.shape[0], 1)), np.zeros((x.shape[0], 1))),
                            dim, opt)
        funcs.append(fun)
    return funcs

CEC2020_NONCONSTRAINED_5D = cec20_func_factory(5)
CEC2020_NONCONSTRAINED_10D = cec20_func_factory(10)
CEC2020_NONCONSTRAINED_15D = cec20_func_factory(15)
CEC2020_NONCONSTRAINED_20D = cec20_func_factory(20)