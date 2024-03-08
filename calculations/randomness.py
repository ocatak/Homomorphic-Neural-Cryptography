import numpy as np
import matplotlib.pyplot as plt

def randomness(rate, curve):
    # Load ciphertexts
    C1 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-1.npy")
    C2 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-2.npy")

    # Calculating the differences
    differences = C1 - C2

    # Calculating the variance and standard deviation of the differences
    variance = np.var(differences)
    std_dev = np.std(differences)

    return variance, std_dev

def get_sd(num_samples, curve):
    std = []
    for rate in num_samples:
        std.append(randomness(rate, curve))
    return std

def plot_evaluation_scores(num_samples, sec224r1, sec256k1, secp256r1, sec384r1, sec521r1):
    plt.figure(figsize=(8, 6))

    # Plot each metric on the graph
    plt.plot(num_samples, sec224r1, 'o-', label='sec224r1')
    plt.plot(num_samples, sec256k1, 's-', label='sec256k1')
    plt.plot(num_samples, secp256r1, '^-', label='secp256r1')
    plt.plot(num_samples, sec384r1, '*-', label='sec384r1')
    plt.plot(num_samples, sec521r1, '+-', label='sec521r1')

    # Adding labels and title
    plt.xlabel('Dropout Rate')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    rate = 0.6
    curve = "secp521r1"
    randomness(rate, curve)

    num_samples = [ 0.6, 0.7]
    sec224r1 = get_sd(num_samples, "secp224r1")
    sec256k1 = get_sd(num_samples, "secp256k1")
    sec256r1 = get_sd(num_samples, "secp256r1")
    sec384r1 = get_sd(num_samples, "secp384r1")
    secp521r1 = get_sd(num_samples, "secp521r1")

    plot_evaluation_scores(num_samples, sec224r1, sec256k1, sec256r1, sec384r1, secp521r1)

