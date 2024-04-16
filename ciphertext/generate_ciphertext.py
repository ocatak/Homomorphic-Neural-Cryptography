import numpy as np
from neural_network.networks_functions import create_networks
from typing import List

def save_generated_ciphertexts(dropout_rates: List[str], curves: List[str], batch_size: int, i: int):
    """Generates a batch of ciphertexts given a dropout rate and curve and saves the first.
    Used to create saved ciphertexts to calculate the mean and standard deviation.

    Args:
        dropout_rates: A list of dropout rates which the model was trained on.
        curves: A list of elliptic curves which the model was trained on.
        batch_size: Number of samples in the dataset.
        i: A number to append to the filename to differentiate between different runs.
    """
    nonce_bits = 64
    p1_batch = np.load("plaintext/p1_batch.npy")
    p2_batch = np.load("plaintext/p2_batch.npy")

    for curve in curves:
        public_arr = np.load(f"key/public_key-{curve}.npy")
        private_arr = np.load(f"key/private_key-{curve}.npy")
        nonce = np.random.rand(batch_size, nonce_bits)
        for rate in dropout_rates:
            alice, _, _, _, _, _, _, _, _, _, _, _ = create_networks(public_arr.shape[1], private_arr.shape[1], rate)
            path_name = f"rate-{rate}-curve-{curve}"
            weights_path = f'weights/weights-{path_name}'
            alice.load_weights(f'{weights_path}/alice_weights.h5')
            cipher1, _ = alice.predict([public_arr, p1_batch, p2_batch, nonce])
            np.save(f"ciphertext/rate-{rate}-curve-{curve}-{i}.npy", cipher1)

if __name__ == "__main__":
    i = 3
    batch_size = 512
    dropout_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    curves = ["secp224r1"]
    save_generated_ciphertexts(dropout_rates, curves, batch_size, i)