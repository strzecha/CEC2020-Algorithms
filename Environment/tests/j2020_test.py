import numpy as np
from implementations.j2020 import j2020Algorithm, j2020Individual

def test_individual_operations():
    i1 = j2020Individual(10, 5, 1)
    i2 = j2020Individual(5, 1, 1)
    i3 = j2020Individual(1, 0, 0)

    i4 = i1 - i2 * 5
    i5 = i2 + i3

    assert i1.x == 10
    assert i4.x == -15
    assert i5.x == 6

def test_j2020_selection():
    x = j2020Individual(np.array([1, 1]), 1, 1)
    u1 = j2020Individual(np.array([2, 2]), 1, 1)
    u2 = j2020Individual(np.array([0, 0]), 1, 1)

    fun = lambda x: x[0] + x[1]

    algorithm = j2020Algorithm()

    x = algorithm.evaluate(x, fun)
    u1 = algorithm.evaluate(u1, fun)
    u2 = algorithm.evaluate(u2, fun)

    x1 = algorithm.selection(x, u1)
    x2 = algorithm.selection(x, u2)

    assert x1 == x
    assert x2 == u2

def test_j2020_best_of():
    x1 = j2020Individual(np.array([0, 0]), 1, 1)
    x2 = j2020Individual(np.array([1, 1]), 1, 1)
    x3 = j2020Individual(np.array([2, 2]), 1, 1)
    fun = lambda x: x[0] + x[1]
    algorithm = j2020Algorithm()

    for x in [x1, x2, x3]:
        x = algorithm.evaluate(x, fun)

    assert algorithm.best_of([x1, x2, x3]) == x1

def test_j2020_crossover(monkeypatch):
    monkeypatch.setattr(np.random, 'random', lambda: 0.1)

    x = j2020Individual(np.array([0, 0]), 1, CR=0.15)
    v = j2020Individual(np.array([1, 2]), 1, 1)
    algorithm = j2020Algorithm(tau1=-1)
    algorithm.dimensionality = 2

    u = algorithm.crossover(x, v)

    assert u.x[0] == 1
    assert u.x[1] == 2
