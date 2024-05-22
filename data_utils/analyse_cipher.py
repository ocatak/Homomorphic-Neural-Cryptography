import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.typing import NDArray

def probabilistic_encryption_analysis(rate: float, curve: str, batch_size: int) -> NDArray[np.float32]:
    """Calculates the standard deviation of five ciphertexts given a dropout rate and curve.
    Used to calculate the standard deviation of the ciphertexts to evaluate the probabilistic encryption.
    
    Args:
        rate: Dropout rate.
        curve: Elliptic curve.
        batch_size: Number of samples in the dataset.
    
    Returns:
        std: Standard deviation of the ciphertexts.
    """
    C1 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-batch-{batch_size}-1.npy")
    C2 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-batch-{batch_size}-2.npy")
    C3 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-batch-{batch_size}-3.npy")
    C4 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-batch-{batch_size}-4.npy")
    C5 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-batch-{batch_size}-5.npy")
    stacked_arrays = np.stack((C1, C2, C3, C4, C5))
    std = np.std(stacked_arrays, axis=0)
    return std[0]

def plot_std_and_mean(dropout_rates: List[float], curve: str, batch_size: int):
    """Plots the mean and standard deviation of the standard deviation of five ciphertexts for a given curve. 
    Saves the plot as a pdf file.

    Args:
        dropout_rates: List of dropout rates.
        curve: Elliptic curve.
        batch_size: Number of samples in the dataset.
    """
    sns.set_theme(style="darkgrid")

    data = []
    for rate in dropout_rates:
        std = probabilistic_encryption_analysis(rate, curve, batch_size)
        for s in std:
            data.append({'value': s, 'type': 'Standard Deviation', 'dropout_rate': rate})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    plt.rc('font', size=12)  
    plt.rc('axes', labelsize=30) 
    plt.rc('xtick', labelsize=25)  
    plt.rc('ytick', labelsize=25)   
    plt.ylim(0, 0.3)
    sns.lineplot(data=df, x='dropout_rate', y='value', hue='type', style='type', markers=True, dashes=False, errorbar='sd', legend=False)
    plt.xlabel('Dropout Rate')
    plt.ylabel('Standard Deviation')
    plt.grid(True)
    plt.savefig(f"pdf-figures/{curve}-std.pdf", bbox_inches='tight')

if __name__ == "__main__":
    rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    curve = "secp224r1"
    batch_size = 1
    plot_std_and_mean(rates, curve, batch_size)