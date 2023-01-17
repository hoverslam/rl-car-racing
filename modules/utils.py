import numpy as np
import matplotlib.pyplot as plt


def moving_average(data: list, window: int) -> np.ndarray:
    """Compute simple moving average.

    Args:
        data (list): List of values.
        window (int): Size of the moving window.

    Returns:
        np.ndarray: Moving average with the same lenght as the input data.
    """
    weights = np.ones(window) / window

    return np.convolve(data, weights, mode="full")[:len(data)]


def plot_results(results: dict, sma_window: int = 100, title: str = "") -> None:
    """Plot score and moving average.

    Args:
        results (dict): Dictionary with episodes and the corresponding scores.
        sma_window (int, optional): Size of the window for the moving average. Defaults to 100.
        title (str, optional): Title of the plot. Defaults to "".
    """
    # Compute simple moving average
    sma = moving_average(results["score"], sma_window)

    # Plot scores and simple moving average
    plt.plot(results["episode"], results["score"], marker=".", linestyle="", markersize=3, alpha=0.3)
    plt.plot(results["episode"], sma, color="red", label=f"SMA{sma_window}")
    plt.title(title)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
