import numpy as np

def midpoint_target(v, x, D, MIN, MAX):
    for j in range(D):
        if v.x[j] < MIN[j]:
            v.x[j] = (x.x[j] + MIN[j]) / 2
        if v.x[j] > MAX[j]:
            v.x[j] = (x.x[j] + MAX[j]) / 2

    return v

def min_max_reflection(v, x, D, MIN, MAX):
    for j in range(D):
        if v.x[j] < MIN[j]:
            v.x[j] = np.minimum(MAX[j], np.maximum(MIN[j], 2 * MIN[j] - x.x[j]))
        if v.x[j] > MAX[j]:
            v.x[j] = np.maximum(MIN[j], np.minimum(MAX[j], 2 * MIN[j] - x.x[j]))

    return v

def projection(v, D, MIN, MAX):
    for j in range(D):
        if v.x[j] < MIN[j]:
            v.x[j] = MIN[j]
        elif v.x[j] > MAX[j]:
            v.x[j] = MAX[j]

    return v

def wrapping(v, D, MIN, MAX):
    for j in range(D):
        if v.x[j] < MIN[j]:
            v.x[j] += (MAX[j] - MIN[j])
        if v.x[j] > MAX[j]:
            v.x[j] -= (MAX[j] - MIN[j])

    return v

def rand_base_1(v, x, D, MIN, MAX):
    for j in range(D):
        r = 0.1 * np.random.rand()

        if v.x[j] > MAX[j]:
            v.x[j] = (1 - r) * MAX[j] + r * x.x[j]
        if v.x[j] < MIN[j]:
            v.x[j] = (1 - r) * MIN[j] + r * x.x[j]

    return v

def rand_base_2(T, D, MIN, MAX):
    r1 = np.random.rand()
    r2 = np.random.rand()
    for v in T:
        for j in range(D):
            if v.x[j] < MIN[j]:
                v.x[j] = MIN[j] + r1 * (MAX[j] - MIN[j])
            if v.x[j] > MAX[j]:
                v.x[j] = MIN[j] + r2 * (MAX[j] - MIN[j])
    return T