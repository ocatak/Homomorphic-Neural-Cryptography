import os
import numpy as np
from neural_network.networks_functions import create_networks
from data_utils.analyse_cipher import plot_std, probabilistic_encryption_analysis

batch_size = 512
nonce_bits = 64
dropout_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
curves = ["secp224r1", "secp256k1", "secp256r1", "secp384r1", "secp521r1"]

p1_batch = np.load("plaintext/p1_batch.npy")
p2_batch = np.load("plaintext/p2_batch.npy")

def decryption_accurancy(cipher, private_arr, nonce, p_batch):
    # Calculate Bob's decryption accuracy
    decrypted = bob.predict([cipher, private_arr, nonce])
    decrypted_bits = np.round(decrypted).astype(int)
    correct_bits = np.sum(decrypted_bits == (p_batch))
    total_bits = np.prod(decrypted_bits.shape)
    return correct_bits / total_bits * 100

results = {}
for curve in curves:
    results[curve] = {}
    results[curve]['std_std'] = []
    results[curve]['mean_std'] = []
    public_arr = np.load(f"key/public_key-{curve}.npy")
    private_arr = np.load(f"key/private_key-{curve}.npy")
    nonce = np.random.rand(batch_size, nonce_bits)
    for rate in dropout_rates:
        results[curve][rate] = {}
        alice, bob, HO_model, eve, _, _, _, _, _, _, _, _ = create_networks(public_arr.shape[1], private_arr.shape[1], rate)
        path_name = f"rate-{rate}-curve-{curve}"
        weights_path = f'weights/weights-{path_name}'
        if not os.path.exists(weights_path):
            continue
        HO_model.load_weights(f'{weights_path}/addition_weights.h5')
        alice.load_weights(f'{weights_path}/alice_weights.h5')
        bob.load_weights(f'{weights_path}/bob_weights.h5')
        eve.load_weights(f'{weights_path}/eve_weights.h5')
        cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])
        cipher3 = HO_model.predict([cipher1, cipher2])
        results[curve][rate]['p1+p2'] = decryption_accurancy(cipher3, private_arr, nonce, p1_batch+p2_batch)
        results[curve][rate]['p1'] = decryption_accurancy(cipher1, private_arr, nonce, p1_batch)
        results[curve][rate]['p2'] = decryption_accurancy(cipher2, private_arr, nonce, p2_batch)
        results[curve][rate]['mean_std'], results[curve][rate]['std_std'] = probabilistic_encryption_analysis(rate, curve)
        results[curve]['std_std'].append(results[curve][rate]['std_std'])
        results[curve]['mean_std'].append(results[curve][rate]['mean_std'])
    
plot_std(dropout_rates, results['secp224r1'], results['secp256k1'], results['secp256r1'], results['secp384r1'], results['secp521r1'])

for curve in results:
    print(curve)
    for rate in results[curve]:
        print(rate)
        print(results[curve][rate])
        print()