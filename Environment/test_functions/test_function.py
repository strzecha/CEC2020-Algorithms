import numpy as np

class TestFunction:
    def __init__(self, ID, fun, dim, global_minimum=None, inequality_constrains_num=0, equality_constraints_num=0):
        self.ID = ID
        self.fun = fun
        self.dimensionality = dim
        self.equality_constraints_num = equality_constraints_num
        self.inequality_constraints_num = inequality_constrains_num
        self.global_minimum = global_minimum

    def evaluate(self, x):
        return self.fun(x)

    def __call__(self, x):
        return self.evaluate(x)