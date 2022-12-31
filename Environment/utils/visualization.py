import matplotlib.pyplot as plt

def draw_graph_evaluations(algorithms):
    colors = ("b", "r", "g", "p")
    for alg, color in zip(algorithms, colors):
        algorithm, name = alg
        FESs = algorithm.FESs
        bests = algorithm.bests_values
        plt.plot(FESs, bests, color, label=name)
    plt.legend()
    plt.show()