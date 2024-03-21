import numpy as np
from neural_network.networks_functions import create_networks

i = 3
batch_size = 1
nonce_bits = 64
dropout_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
curves = ["secp224r1"]

p1_batch = np.load("plaintext/p1.npy")
p2_batch = np.load("plaintext/p2.npy")

for curve in curves:
    public_arr = np.load(f"key/public_key-{curve}-one.npy")
    private_arr = np.load(f"key/private_key-{curve}-one.npy")
    nonce = np.random.rand(batch_size, nonce_bits)
    for rate in dropout_rates:
        alice, _, _, _, _, _, _, _, _, _, _, _ = create_networks(public_arr.shape[1], private_arr.shape[1], rate)
        path_name = f"rate-{rate}-curve-{curve}"
        weights_path = f'weights/weights-{path_name}'
        alice.load_weights(f'{weights_path}/alice_weights.h5')
        cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])
        print(cipher1)
        np.save(f"ciphertext/one_batch/rate-{rate}-curve-{curve}-{i}.npy", cipher1)
