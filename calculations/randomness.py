
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def probabilistic_encryption_analysis(rate: float, curve: str) -> Tuple[float, float]:
    """Calculates the mean and standard deviation of five ciphertexts given a dropout rate and curve.
    Used to calculate the mean and standard deviation of the ciphertexts to compare the probabilistic encryption.
    
    Args:
        rate: Dropout rate.
        curve: Elliptic curve.
    
    Returns:
        mean_std: Mean of the standard deviation of the ciphertexts.
        std_std: Standard deviation of the standard deviation of the ciphertexts.
    """
    C1 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-1.npy")
    C2 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-2.npy")
    C3 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-3.npy")
    C4 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-4.npy")
    C5 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-5.npy")
    stacked_arrays = np.stack((C1, C2, C3, C4, C5))
    std = np.std(stacked_arrays, axis=0)
    return std[0]

def plot_std_and_mean(curve):
    """Plots the mean and standard deviation of the standard deviation of five ciphertexts for a given curve. 
    Saves the plot as a pdf file.

    Args:
        curve: Elliptic curve.
    """
    # Initialize Seaborn theme
    sns.set_theme(style="darkgrid")

    # Define the dropout rates
    dropout_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Initialize lists to store the data
    data = []

    # Loop through the dropout rates, call the function, and append data
    for rate in dropout_rates:
        std = probabilistic_encryption_analysis(rate, curve)
        # Append standard deviation values
        for s in std:
            data.append({'value': s, 'type': 'mean of std', 'dropout_rate': rate})

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.rc('font', size=12)  # controls default text size
    plt.rc('axes', titlesize=16)  # fontsize of the axes title
    plt.rc('axes', labelsize=25)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=25)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=25)  # fontsize of the tick labels
    plt.rc('legend', fontsize=18)  # fontsize of the legend
    plt.ylim(0, 0.3)
    sns.lineplot(data=df, x='dropout_rate', y='value', hue='type', style='type', markers=True, dashes=False, errorbar='sd')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Value')
    plt.legend(title='Type')
    plt.grid(True)
    plt.savefig(f"pdf-figures/{curve}-std-sns.pdf", bbox_inches='tight')

if __name__ == "__main__":
    rate = 0.01
    curve = "secp224r1"
    print(probabilistic_encryption_analysis(rate, curve)[0].shape)