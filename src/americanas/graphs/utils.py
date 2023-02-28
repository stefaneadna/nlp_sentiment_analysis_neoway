import matplotlib.pyplot as plt
import seaborn as sns


def matrix(matrix):
    """Plota uma matriz de confusão

    Args:
        matrix (array): Um Array referente a uma matriz de confusão de 2x2
    """
    plt.figure(figsize=(5,5))
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="", cmap="Blues")
    ax.set_xticklabels(['Ruim', 'Bom'])
    ax.set_yticklabels(['Ruim', 'Bom'])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Reviews")