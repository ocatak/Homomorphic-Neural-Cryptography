import os
import numpy as np
from neural_network.networks import create_networks
from data_utils.analyse_cipher import plot_std_and_mean, probabilistic_encryption_analysis
from tensorflow.keras.models import Model

batch_size = 1
nonce_bits = 64
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
curves = ["secp224r1", "secp256r1"]

p1_batch = np.load(f"plaintext/p1-{batch_size}.npy")
p2_batch = np.load(f"plaintext/p2-{batch_size}.npy")

def decryption_accurancy(bob: Model, cipher: np.ndarray, private_arr: np.ndarray, nonce: int, p_batch: np.ndarray) -> float:
    """Calculates the decryption accuracy of Bob.
    
    Args:
        bob: Bob model.
        cipher: Ciphertext.
        private_arr: Private key array.
        nonce: Nonce.
        p_batch: Plaintext batch.

    Returns:
        The decryption accuracy of Bob.
    """
    # Calculate Bob's decryption accuracy
    decrypted = bob.predict([cipher, private_arr, nonce])
    decrypted_bits = np.round(decrypted).astype(int)
    correct_bits = np.sum(decrypted_bits == (p_batch))
    total_bits = np.prod(decrypted_bits.shape)
    return correct_bits / total_bits * 100

results = {}
for curve in curves:
    results[curve] = {}
    public_arr = np.load(f"key/public_key-{curve}-{batch_size}.npy")
    private_arr = np.load(f"key/private_key-{curve}-{batch_size}.npy")
    nonce = np.random.rand(batch_size, nonce_bits)
    for rate in dropout_rates:
        results[curve][rate] = {}
        alice, bob, HO_model_addition, eve, _, _, _, _, _, _, c3_bits, _, HO_model_multiplication = create_networks(public_arr.shape[1], private_arr.shape[1], rate)
        path_name = f"ma-rate-{rate}-curve-{curve}"
        weights_path = f'weights/weights-{path_name}'
        if not os.path.exists(weights_path):
            continue
        HO_model_addition.load_weights(f'{weights_path}/addition_weights.h5')
        HO_model_multiplication.load_weights(f'{weights_path}/multiplication_weights.h5')
        alice.load_weights(f'{weights_path}/alice_weights.h5')
        bob.load_weights(f'{weights_path}/bob_weights.h5')
        eve.load_weights(f'{weights_path}/eve_weights.h5')
        cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])
        operation_a = np.zeros((batch_size, c3_bits))
        operation_m = np.ones((batch_size, c3_bits))
        cipher3_a = HO_model_addition.predict([operation_a, cipher1, cipher2])
        cipher3_m = HO_model_multiplication.predict([operation_m, cipher1, cipher2])
        results[curve][rate]['p1+p2'] = decryption_accurancy(bob, cipher3_a, private_arr, nonce, p1_batch+p2_batch)
        results[curve][rate]['p1*p2'] = decryption_accurancy(bob, cipher3_m, private_arr, nonce, p1_batch*p2_batch)
        results[curve][rate]['p1'] = decryption_accurancy(bob, cipher1, private_arr, nonce, p1_batch)
        results[curve][rate]['p2'] = decryption_accurancy(bob, cipher2, private_arr, nonce, p2_batch)
        std = probabilistic_encryption_analysis(rate, curve, batch_size)
        results[curve][rate]['std_std'] = np.std(std)
        results[curve][rate]['mean_std'] = np.mean(std)
    plot_std_and_mean(dropout_rates, curve, batch_size)

for curve in results:
    print(curve)
    for rate in results[curve]:
        print(rate)
        print(results[curve][rate])
        print()