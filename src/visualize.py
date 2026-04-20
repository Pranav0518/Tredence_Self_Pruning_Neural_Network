import matplotlib.pyplot as plt
import seaborn as sns
import torch
from model import PrunableLinear

def plot_tradeoff(results):
    sparsity = [r[2] for r in results]  # extract sparsity values
    accuracy = [r[1] for r in results]  # extract accuracy values

    plt.figure()
    plt.plot(sparsity, accuracy, marker='o')  # trade-off curve

    for lam, acc, sp in results:
        plt.text(sp, acc, f"λ={lam}")  # annotate each point with lambda

    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Sparsity Trade-off")
    plt.grid()
    plt.show()


def plot_gate_distribution(model):
    gates = []

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            # collect all gate values across prunable layers
            gates.extend(torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten())

    plt.figure()
    sns.histplot(gates, bins=50)  # visualize distribution of gate values
    plt.title("Gate Distribution")
    plt.show()