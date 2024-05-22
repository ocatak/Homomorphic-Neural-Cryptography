import numpy as np
from neural_network.networks import create_networks
from tensorflow.keras.models import Model

dropout_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
curves = ["secp224r1", "secp256k1", "secp256r1", "secp384r1", "secp521r1"]

batch_size = 448
nonce_bits = 64

p1_batch = np.load(f"plaintext/p1-{batch_size}.npy")
p2_batch = np.load(f"plaintext/p2-{batch_size}.npy")

def decryption_accurancy(eve: Model, cipher: np.ndarray, public_arr: np.ndarray, nonce: int, p_batch: np.ndarray) -> float:
    """Calculates the decryption accuracy of Bob.
    
    Args:
        eve: Eve model.
        cipher: Ciphertext.
        public_arr: Public key array.
        nonce: Nonce.
        p_batch: Plaintext batch.

    Returns:
        The decryption accuracy of Eve.
    """
    # Calculate Bob's decryption accuracy
    decrypted = eve.predict([cipher, public_arr, nonce])
    decrypted_bits = np.round(decrypted).astype(int)
    correct_bits = np.sum(decrypted_bits == (p_batch))
    total_bits = np.prod(decrypted_bits.shape)
    return correct_bits / total_bits * 100

addition = []
mulitplication = []
p1_decrypted = []
p2_decrypted = []
for curve in curves:
    public_arr = np.load(f"key/public_key-{curve}-{batch_size}.npy")
    private_arr = np.load(f"key/private_key-{curve}-{batch_size}.npy")
    nonce = np.random.rand(batch_size, nonce_bits)
    for rate in dropout_rates:
        alice, bob, HO_model_addition, eve, _, _, _, _, _, _, c3_bits, _, HO_model_multiplication  = create_networks(public_arr.shape[1], private_arr.shape[1], rate)
        path_name = f"ma-rate-{rate}-curve-{curve}-extra-out"
        weights_path = f'weights/weights-{path_name}'
        HO_model_addition.load_weights(f'{weights_path}/addition_weights.h5')
        HO_model_multiplication.load_weights(f'{weights_path}/multiplication_weights.h5')
        alice.load_weights(f'{weights_path}/alice_weights.h5')
        eve.load_weights(f'{weights_path}/eve_weights.h5')
        cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])
        operation_a = np.zeros((batch_size, c3_bits))
        operation_m = np.ones((batch_size, c3_bits))
        cipher3_a = HO_model_addition.predict([operation_a, cipher1, cipher2])
        cipher3_m = HO_model_multiplication.predict([operation_m, cipher1, cipher2])
        addition.append(decryption_accurancy(eve, cipher3_a, public_arr, nonce, p1_batch+p2_batch))
        mulitplication.append(decryption_accurancy(eve, cipher3_m, public_arr, nonce, p1_batch*p2_batch))
        p1_decrypted.append(decryption_accurancy(eve, cipher1, public_arr, nonce, p1_batch))
        p2_decrypted.append(decryption_accurancy(eve, cipher2, public_arr, nonce, p2_batch))

print(f"Addition: {addition}")
print(f"Multiplication: {mulitplication}")
print(f"P1: {p1_decrypted}")
print(f"P2: {p2_decrypted}")

print(f"Addition: {np.mean(addition)}")
print(f"Multiplication: {np.mean(mulitplication)}")
print(f"P1: {np.mean(p1_decrypted)}")
print(f"P2: {np.mean(p2_decrypted)}")
