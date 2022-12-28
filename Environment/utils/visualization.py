import matplotlib.pyplot as plt

def draw_graph_evaluations(algorithms):
    for algorithm, name in algorithms:
        bests = algorithm.bests_values
        plt.plot(range(1, len(bests) + 1), bests, "b", label=name)
    plt.legend()
    plt.show()