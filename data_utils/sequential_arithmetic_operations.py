from neural_network.networks import create_networks
import numpy as np
from data_utils.accuracy import decryption_accurancy
from typing import Tuple


def sequential_arithmetic_operations(rate: float, batch_size: int, curve: str) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Test to verify that the system is a somewhat homomorphic encryption system

    Args:
        rate: The dropout rate
        batch_size: The batch size
        curve: The curve
    """
    test_type = f"ma-rate-{rate}-curve-{curve}"

    public_arr = np.load(f"key/public_key-{curve}-{batch_size}.npy")
    private_arr = np.load(f"key/private_key-{curve}-{batch_size}.npy")

    alice, bob, HO_model_addition, _, _, _, p1_bits, _, p2_bits, _, _, nonce_bits, HO_model_multiplication = create_networks(public_arr.shape[1], private_arr.shape[1], rate)

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

    # HO multiplies the messages
    op_m = np.ones((cipher1.shape))
    cipher6 = HO_model_multiplication.predict([op_m, cipher1, cipher2])
    cipher7 = HO_model_multiplication.predict([op_m, cipher3, cipher4])

    # HO adds and multiplies the messages
    cipher8 = HO_model_multiplication.predict([op_m, cipher4, cipher3])
    cipher9 = HO_model_addition.predict([op_a, cipher6, cipher3])

    p5 = decryption_accurancy(bob, cipher5, private_arr, nonce, p1_batch+p2_batch+p3_batch)
    p7 = decryption_accurancy(bob, cipher7, private_arr, nonce, p1_batch*p2_batch*p3_batch)
    p8 = decryption_accurancy(bob, cipher8, private_arr, nonce, (p1_batch+p2_batch)*p3_batch)
    p9 = decryption_accurancy(bob, cipher9, private_arr, nonce, p1_batch*p2_batch+p3_batch)
    return p5, p7, p8, p9

if __name__ == "__main__":
    rate = 0.2
    batch_size = 448
    curve = "secp256r1"
    sequential_arithmetic_operations(rate, batch_size, curve)