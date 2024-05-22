import numpy as np
from neural_network.networks import create_networks
from typing import List

def save_generated_ciphertexts(dropout_rates: List[float], curves: List[str], batch_size: int, i: int):
    """Generates a batch of ciphertexts given a dropout rate and curve and saves them.
    Used to create saved ciphertexts to calculate the mean and standard deviation.

    Args:
        dropout_rates: A list of dropout rates which the model was trained on.
        curves: A list of elliptic curves which the model was trained on.
        batch_size: Number of samples in the dataset.
        i: A number to append to the filename to differentiate between different ciphertexts.
    """
    nonce_bits = 64
    p1_batch = np.load(f"plaintext/p1-{batch_size}.npy")
    p2_batch = np.load(f"plaintext/p2-{batch_size}.npy")
    for curve in curves:
        public_arr = np.load(f"key/public_key-{curve}-{batch_size}.npy")
        private_arr = np.load(f"key/private_key-{curve}-{batch_size}.npy")
        nonce = np.random.rand(batch_size, nonce_bits)
        for rate in dropout_rates:
            alice, _, _, _, _, _, _, _, _, _, _, _, _= create_networks(public_arr.shape[1], private_arr.shape[1], rate)
            path_name = f"ma-rate-{rate}-curve-{curve}-extra-out"
            weights_path = f'weights/weights-{path_name}'
            alice.load_weights(f'{weights_path}/alice_weights.h5')
            cipher1, _ = alice.predict([public_arr, p1_batch, p2_batch, nonce])
            np.save(f"ciphertext/rate-{rate}-curve-{curve}-batch-{batch_size}-{i}.npy", cipher1)

if __name__ == "__main__":
    i = 1
    batch_size = 1
    dropout_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    curves = ["secp224r1", "secp256k1", "secp256r1", "secp384r1", "secp521r1"]
    save_generated_ciphertexts(dropout_rates, curves, batch_size, i)