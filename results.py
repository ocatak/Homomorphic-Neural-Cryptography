import os
import numpy as np
from neural_network.networks import create_networks
from data_utils.analyse_cipher import plot_std_and_mean, probabilistic_encryption_analysis
from data_utils.accuracy import decryption_accurancy, HO_accuracy
from data_utils.sequential_arithmetic_operations import sequential_arithmetic_operations
from typing import List

def print_results(results: dict):
    """
    Print the results of a dictionary

    Args:
        results: The dictionary of results
    """
    for curve in results:
        print(f"Curve: {curve}")
        for rate in results[curve]:
            print(f"Rate {rate}: {results[curve][rate]}")
        print()

def get_accuracy(batch_size: int, nonce_bits: int, dropout_rates: List[float], curves: List[str]):
    """
    Prints the decryption accuracy of Bob and Eve, including the accuracy of the HO model
    
    Args:
        batch_size: Number of samples in the dataset
        nonce_bits: The number of bits in the nonce
        dropout_rates: The dropout rates
        curves: The curves
    """
    p1_batch = np.load(f"plaintext/p1-{batch_size}.npy")
    p2_batch = np.load(f"plaintext/p2-{batch_size}.npy")
    results = {}
    addition = []
    mulitplication = []
    p1_decrypted = []
    p2_decrypted = []
    HO_addition = []
    HO_multiplication = []
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
            # Calculate HO model's accuracy
            HO_addition.append(HO_accuracy(cipher3_a, cipher1+cipher2))
            HO_multiplication.append(HO_accuracy(cipher3_m, cipher1*cipher2))
            # Calculate Bob's decryption accuracy
            results[curve][rate]['p1+p2'] = decryption_accurancy(bob, cipher3_a, private_arr, nonce, p1_batch+p2_batch)
            results[curve][rate]['p1*p2'] = decryption_accurancy(bob, cipher3_m, private_arr, nonce, p1_batch*p2_batch)
            results[curve][rate]['p1'] = decryption_accurancy(bob, cipher1, private_arr, nonce, p1_batch)
            results[curve][rate]['p2'] = decryption_accurancy(bob, cipher2, private_arr, nonce, p2_batch)
            # Calculate Eve's decryption accuracy
            addition.append(decryption_accurancy(eve, cipher3_a, public_arr, nonce, p1_batch+p2_batch))
            mulitplication.append(decryption_accurancy(eve, cipher3_m, public_arr, nonce, p1_batch*p2_batch))
            p1_decrypted.append(decryption_accurancy(eve, cipher1, public_arr, nonce, p1_batch))
            p2_decrypted.append(decryption_accurancy(eve, cipher2, public_arr, nonce, p2_batch))

    print("Bob's decryption accuracy: ")
    print_results(results)

    print(f"Eve's average decryption accuracy: ")
    print(f"Addition: {np.round(np.mean(addition), 2)}")
    print(f"Multiplication: {np.round(np.mean(mulitplication), 2)}")
    print(f"P1: {np.round(np.mean(p1_decrypted), 2)}")
    print(f"P2: {np.round(np.mean(p2_decrypted), 2)}\n")

    print(f"HO Addition model's average accuracy: {np.round(np.mean(HO_addition), 2)}")
    print(f"HO Multiplication model's average accuracy: {np.round(np.mean(HO_multiplication), 2)} \n")

def get_std_results(rates: List[float], curves: List[str], batch_size: int = 1):
    """
    Get the standard deviation results of the probabilistic encryption
    
    Args:
        rates: The dropout rates
        curves: The curves
        batch_size: Number of samples in the dataset
    """
    results = {}
    for curve in curves:
        results[curve] = {}
        for rate in rates:
            results[curve][rate] = {}
            std = probabilistic_encryption_analysis(rate, curve, batch_size)
            results[curve][rate]['std_std'] = round(np.std(std), 4)
            results[curve][rate]['mean_std'] = round(np.mean(std), 4)
        plot_std_and_mean(dropout_rates, curve, batch_size)
    print("Standard deviation results: ")
    print_results(results)

def get_decryption_accuracy_on_sequential_operations(rate: float, batch_size: int, curve: str):
    """
    Get Bob's decryption accuracy on sequential operations performed by the HO networks
    
    Args:
        rate: The dropout rate
        batch_size: The batch size
        curve: The curve
        
    Returns:
        The decryption accuracy of Bob
    """
    p5, p7, p8, p9 = sequential_arithmetic_operations(rate, batch_size, curve)
    print(f"Decryption accuracy P1+P2+P3: {p5}")
    print(f"Decryption accuracy P1*P2*P3: {p7}")
    print(f"Decryption accuracy (P1+P2)*P3: {p8}")
    print(f"Decryption accuracy P1*P2+P3: {p9}")


if __name__ == "__main__":
    batch_size = 448
    nonce_bits = 64
    dropout_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    curves = ["secp224r1", "secp256k1", "secp256r1", "secp384r1", "secp521r1"]
    get_accuracy(batch_size, nonce_bits, dropout_rates, curves)
    get_std_results(dropout_rates, curves)
    get_decryption_accuracy_on_sequential_operations(0.2, batch_size, "secp256r1")