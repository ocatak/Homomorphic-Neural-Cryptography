from neural_network.networks import create_networks
import numpy as np
from key.EllipticCurve import set_curve
from tensorflow.keras.models import Model

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

curve = set_curve("secp256r1")

rate = 0.2

batch_size = 448
test_type = f"ma-rate-{rate}-curve-secp256r1-extra-out"

public_arr = np.load(f"key/public_key-{curve.name}-448.npy")
private_arr = np.load(f"key/private_key-{curve.name}-448.npy")

alice, bob, HO_model_addition, eve, _, _, p1_bits, _, p2_bits, _, _, nonce_bits, HO_model_multiplication = create_networks(public_arr.shape[1], private_arr.shape[1], rate)

HO_a_weights_path = f'weights/weights-{test_type}/addition_weights.h5'
HO_m_weights_path = f'weights/weights-{test_type}/multiplication_weights.h5'
alice_weights_path = f'weights/weights-{test_type}/alice_weights.h5'
bob_weights_path = f'weights/weights-{test_type}/bob_weights.h5'

HO_model_addition.load_weights(HO_a_weights_path)
HO_model_multiplication.load_weights(HO_m_weights_path)
alice.load_weights(alice_weights_path)
bob.load_weights(bob_weights_path)

nonce = np.random.rand(batch_size, nonce_bits)

p1_batch = np.random.randint(
    0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
p2_batch = np.random.randint(
    0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
p3_batch = np.random.randint(
    0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
p33_batch = np.random.randint(
    0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')

# Alice encrypts the message
cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])
cipher3, _ = alice.predict([public_arr, p3_batch, p33_batch, nonce])

# HO adds the messages
op_a = np.zeros((cipher1.shape))
cipher4 = HO_model_addition.predict([op_a, cipher1, cipher2])
cipher5 = HO_model_addition.predict([op_a, cipher3, cipher4])

computed_cipher = cipher1 + cipher2 + cipher3
tolerance = 1e-4
correct_elements = np.sum(np.abs(computed_cipher - cipher5) <= tolerance)
total_elements = np.prod(cipher5.shape)
accuracy_percentage_add = (correct_elements / total_elements) * 100
print(f"Addition accuracy: {accuracy_percentage_add}")

# HO multiplies the messages
op_m = np.ones((cipher1.shape))
cipher6 = HO_model_multiplication.predict([op_m, cipher1, cipher2])
cipher7 = HO_model_multiplication.predict([op_m, cipher3, cipher4])

computed_cipher = cipher1 * cipher2 * cipher3
tolerance = 1e-4
correct_elements = np.sum(np.abs(computed_cipher - cipher7) <= tolerance)
total_elements = np.prod(cipher7.shape)
accuracy_percentage_mul = (correct_elements / total_elements) * 100
print(f"Multiplication accuracy: {accuracy_percentage_mul}")

cipher8 = HO_model_multiplication.predict([op_m, cipher4, cipher3])
cipher9 = HO_model_addition.predict([op_a, cipher6, cipher3])

print(f"Decryption accuracy P1: {decryption_accurancy(bob, cipher1, private_arr, nonce, p1_batch)}")
print(f"Decryption accuracy P3: {decryption_accurancy(bob, cipher3, private_arr, nonce, p3_batch)}")
print(f"Decryption accuracy P1+P2: {decryption_accurancy(bob, cipher4, private_arr, nonce, p1_batch+p2_batch)}")
print(f"Decryption accuracy P1+P2+P3: {decryption_accurancy(bob, cipher5, private_arr, nonce, p1_batch+p2_batch+p3_batch)}")
print(f"Decryption accuracy P1*P2: {decryption_accurancy(bob, cipher6, private_arr, nonce, p1_batch*p2_batch)}")
print(f"Decryption accuracy P1*P2*P3: {decryption_accurancy(bob, cipher7, private_arr, nonce, p1_batch*p2_batch*p3_batch)}")
print(f"Decryption accuracy (P1+P2)*P3: {decryption_accurancy(bob, cipher8, private_arr, nonce, (p1_batch+p2_batch)*p3_batch)}")
print(f"Decryption accuracy P1*P2+P3: {decryption_accurancy(bob, cipher9, private_arr, nonce, p1_batch*p2_batch+p3_batch)}")
