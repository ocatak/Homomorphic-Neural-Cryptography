import numpy as np

def randomness(rate, curve):
    # Load ciphertexts
    C1 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-1.npy")
    C2 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-2.npy")

    # Calculating the differences
    differences = C1 - C2

    # Calculating the variance and standard deviation of the differences
    variance = np.var(differences)
    std_dev = np.std(differences)

    print(variance, std_dev)

    return std_dev

if __name__ == "__main__":
    rate = 0.2
    curve = "secp224r1"

