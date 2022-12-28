import benchmark_functions as bf

def fun1D(x):
    # global minimum: min(fun1D(x)) = -21.501 at x = (1.5092)
    return 2 * x[0] ** 4 - 3 * x[0] ** 3 - 7 * x[0] - 11

def fun2D(x):
    # global minimum: min(fun2D(x)) = 2 at x = (-1, 0)
    return 2 * x[0] ** 2 + 4 * x[0] + 2 * x[1] ** 6 + 4

def fun3D(x):
    # global minimum: min(fun3D(x)) = -10.2878 at x = (0.572357, 0, -0.000113599)
    return (4 * x[0] ** 4) - (3 * x[0]) + (7 * x[1] ** 6) + (x[2] ** 4) - (9)

def hypersphere2D(x):
    # global minimum: min(hypersphere2D(x)) = 0 at x = (0, 0)
    hypersphere = bf.Hypersphere(n_dimensions=2)
    return hypersphere(x)

def hypersphere5D(x):
    # global minimum: min(hypersphere5D(x)) = 0 at x = (0, 0, 0, 0, 0)
    hypersphere = bf.Hypersphere(n_dimensions=5)
    return hypersphere(x)

def hypersphere10D(x):
    # global minimum: min(hypersphere10D(x)) = 0 at x = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    hypersphere = bf.Hypersphere(n_dimensions=10)
    return hypersphere(x)

def easom2D(x):
    # global minimum: min(easom2D(x)) = -1 at x = (3.14, 3.14)
    easom = bf.Easom()
    return easom(x)

def schwefel2D(x):
    schwefel = bf.Schwefel(n_dimensions=2)
    return schwefel(x)

def himmelblau2D(x):
    himmelblau = bf.Himmelblau()
    return himmelblau(x)

