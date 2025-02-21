import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(rate: float, curve: str):
    """Plots the loss of the ABHE, Bob and Eve models during training and save the plot as a pdf file.

    Args:
        rate: Dropout rate.
        curve: Elliptic curve.
    """
    plt.figure(figsize=(10, 6))
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.rc('legend', fontsize=25)
    plt.ylim(0, 16)

    df = pd.read_csv(f'/Users/espensele/Desktop/Master/MasterOppg/Homomorphic-Neural-Cryptography/dataset/ma-rate-{rate}-curve-{curve}.csv')

    x = list(range(0, len(df)))

    y_ab = df['ABloss']
    y_bob = df['Bobloss']
    y_eve = df['Eveloss']

    plt.plot(x, y_ab, color='blue', linewidth=1, label='ABHE')
    plt.plot(x, y_bob, color='green', linewidth=1, label='Bob')
    plt.plot(x, y_eve, color='orange', linewidth=1, label='Eve')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Iterations') 
    plt.ylabel('Loss') 
    plt.legend()
    plt.savefig(f"../figures/training_loss.pdf", bbox_inches='tight')

if __name__ == "__main__":
    plot_loss(0.3, "secp224r1-2")