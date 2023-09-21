import numpy as np

def get_cec2020_nonconstrained_optimum():
    optimum = [100, 1100, 700, 1900, 1700, 1600, 2100, 2200, 2400, 2500]
    return optimum

def get_cec2020_constrained_optimum():
    optimum = np.array([
        1.8931162966E+02,
        7.0490369540E+03,
        -4.5291197395E+03,
        -3.8826043623E-01,
        -4.0000560000E+02,
        1.8638304088E+00,
        2.1158627569E+00,
        2.0000000000E+00,
        2.5576545740E+00,
        1.0765430833E+00,
        9.9238463653E+01,
        2.9248305537E+00,
        2.6887000000E+04,
        5.3638942722E+04,
        2.9944244658E+03,
        3.2213000814E-02,
        1.2665232788E-02,
        5.8853327736E+03,
        1.6702177263E+00,
        2.6389584338E+02,
    ])

    return optimum

def initialize_cec2020_nonconstrained_parameters(D):   
    if D == 5:
        FES_Max = 50000
    elif D == 10:
        FES_Max = 1000000
    elif D == 15:
        FES_Max = 3000000
    elif D == 20:
        FES_Max = 10000000
        
    MIN = np.array([-100] * D)
    MAX = np.array([100] * D)
    
    return FES_Max, MIN, MAX

def get_dim_and_constraints_cec20_constrained(fun_num):
    par = {}
    d = [9, 11, 7, 6, 9, 38, 48, 2, 3, 3, 7, 7, 5, 10, 7, 14, 3, 4, 4, 2, 5, 9, 5, 7, 4]
    D = d[fun_num-1]
    gn = [0, 0, 14, 1, 2, 0, 0, 2, 1, 3, 4, 9, 3, 10, 11, 15, 4, 4, 5, 3, 8, 10, 8, 7, 7]
    hn = [8, 9, 0, 4, 4, 32, 38, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0]
    G = gn[fun_num-1]
    H = hn[fun_num-1]

    
    return D, G, H

def initialize_cec2020_constrained_parameters(fun_num):
    dims = [9, 11, 7, 6, 9, 38, 48, 2, 3, 3, 7, 7, 5, 10, 7, 14, 3, 4, 4, 2, 5, 9, 5, 7, 4]
    D = dims[fun_num - 1]
    if D <= 10:
        FES_Max = 100000
    elif D > 10 and D <= 30:
        FES_Max = 200000
    elif D > 30 and D <= 50:
        FES_Max = 400000
    elif D > 50 and D <= 150:
        FES_Max = 800000
    else:
        FES_Max = 1000000

    # Range
    xmin1 = np.array([0, 0, 0, 0, 1000, 0, 100, 100, 100])
    xmax1 = np.array([10, 200, 100, 200, 2000000, 600, 600, 600, 900])
    xmin2 = np.array([10 ** 4, 10 ** 4, 10 ** 4, 0, 0, 0, 100, 100, 100, 100, 100])
    xmax2 = np.array([0.819 * 10 ** 6, 1.131 * 10 ** 6, 2.05 * 10 ** 6, 0.05074, 0.05074, 0.05074, 200, 300, 300, 300, 400])
    xmin3 = np.array([1000, 0, 2000, 0, 0, 0, 0])
    xmax3 = np.array([2000, 100, 4000, 100, 100, 20, 200])
    xmin4 = np.array([0, 0, 0, 0, 1e-5, 1e-5])
    xmax4 = np.array([1, 1, 1, 1, 16, 16])
    xmin5 = -0 * np.ones(D)
    xmax5 = np.array([100, 200, 100, 100, 100, 100, 200, 100, 200])
    xmin6 = 0 * np.ones(D)
    xmax6 = np.array([90, 150, 90, 150, 90, 90, 150, 90, 90, 90, 150, 150, 90, 90, 150, 90, 150, 90, 150, 90, 1, 1.2, 1, 1, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 1.2, 0.5, 1.2, 1.2, 0.5, 1.2, 1.2])
    xmin7 = -0 * np.ones(D)
    xmin7 = -0 * np.ones(dims[6])
    xmin7[[23, 25, 27, 30]] = 0.849999
    xmax7 = 1 * np.ones(dims[6])
    xmax7[3] = 140
    xmax7[[24, 26, 31, 34, 36, 28]] = 30
    xmax7[[1, 2, 4, 12, 13, 14]] = 90
    xmax7[[0, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19]] = 35
    xmin8 = np.array([0, -0.51])
    xmax8 = np.array([1.6, 1.49])
    xmin9 = np.array([0.5, 0.5, -0.51])
    xmax9 = np.array([1.4, 1.4, 1.49])
    xmin10 = np.array([0.2, -2.22554, -0.51])
    xmax10 = np.array([1, -1, 1.49])
    xmin11 = np.array([0, 0, 0, 0, -0.51, -0.51, 0])
    xmax11 = np.array([20, 20, 10, 10, 1.49, 1.49, 40])
    xmin12 = np.array([0, 0, 0, -0.51, -0.51, -0.51, -0.51])
    xmax12 = np.array([100, 100, 100, 1.49, 1.49, 1.49, 1.49])
    xmin13 = np.array([27, 27, 27, 77.51, 32.51])
    xmax13 = np.array([45, 45, 45, 102.49, 45.49])
    xmin14 = np.array([0.51, 0.51, 0.51, 250, 250, 250, 6, 4, 40, 10])
    xmax14 = np.array([3.49, 3.49, 3.49, 2500, 2500, 2500, 20, 16, 700, 450])
    xmin15 = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5])
    xmax15 = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
    xmin16 = 0.001 * np.ones(D)
    xmax16 = 5 * np.ones(D)
    xmin17 = np.array([0.05, 0.25, 2.00])
    xmax17 = np.array([2, 1.3, 15.0])
    xmin18 = np.array([0.51, 0.51, 10, 10])
    xmax18 = np.array([99.49, 99.49, 200, 200])
    xmin19 = np.array([0.125, 0.1, 0.1, 0.1])
    xmax19 = np.array([2, 10, 10, 2])
    xmin20 = np.zeros(D)
    xmax20 = np.ones(D)

    xmins = {
        1: xmin1, 2: xmin2, 3: xmin3, 4: xmin4, 5: xmin5,
        6: xmin6, 7: xmin7, 8: xmin8, 9: xmin9, 10: xmin10,
        11: xmin11, 12: xmin12, 13: xmin13, 14: xmin14, 15: xmin15,
        16: xmin16, 17: xmin17, 18: xmin18, 19: xmin19, 20: xmin20,
    }

    xmaxs = {
        1: xmax1, 2: xmax2, 3: xmax3, 4: xmax4, 5: xmax5,
        6: xmax6, 7: xmax7, 8: xmax8, 9: xmax9, 10: xmax10,
        11: xmax11, 12: xmax12, 13: xmax13, 14: xmax14, 15: xmax15,
        16: xmax16, 17: xmax17, 18: xmax18, 19: xmax19, 20: xmax20,
    }

    MAX = xmaxs[fun_num]
    MIN = xmins[fun_num]

    return FES_Max, MIN, MAX, D