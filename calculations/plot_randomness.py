import matplotlib.pyplot as plt
from randomness import randomness

def plot_evaluation_scores(num_samples, sec224r1, sec256k1, secp256r1, sec384r1):
    plt.figure(figsize=(8, 6))

    # Plot each metric on the graph
    plt.plot(num_samples, sec224r1, 'o-', label='sec224r1')
    plt.plot(num_samples, sec256k1, 's-', label='sec256k1')
    plt.plot(num_samples, secp256r1, '^-', label='secp256r1')
    plt.plot(num_samples, sec384r1, '*-', label='sec384r1')

    # Adding labels and title
    plt.xlabel('Dropout Rate')
    plt.ylabel('Standard Deviation')
    
    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()


def get_sd(num_samples, curve):
    std = []
    for rate in num_samples:
        std.append(randomness(rate, curve))
    return std

# Example usage:
num_samples = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

sec224r1 = get_sd(num_samples, "secp224r1")
sec256k1 = get_sd(num_samples, "secp256k1")
sec256r1 = get_sd(num_samples, "secp256r1")
sec384r1 = get_sd(num_samples, "secp384r1")

plot_evaluation_scores(num_samples, sec224r1, sec256k1, sec256r1, sec384r1)