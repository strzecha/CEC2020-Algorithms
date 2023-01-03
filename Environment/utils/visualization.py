import matplotlib.pyplot as plt

def draw_graph_evaluations(algorithms):
    colors = ("b", "r", "g", "y", "m")
    for alg, color in zip(algorithms, colors):
        algorithm, name = alg
        FESs = algorithm.FESs
        bests = algorithm.bests_values
        plt.plot(FESs, bests, color, label=name)
    plt.xlabel("FES")
    plt.ylabel("fun value")
    plt.legend()
    plt.show()