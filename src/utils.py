import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(predicted, path, original):
    """plots and save input and reconstructed as pdf

    Args:
        sample (list(tuple)): list of (input , reconstructed) to be saved 
        prefix (str): train, val or test prefix
        path (str): save path
    """
    plt.figure()
    plt.plot(original, label='Original')
    plt.plot(predicted, label='predicted')
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.close()
