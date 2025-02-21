import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from tensorflow.keras.models import Model
from numpy.typing import NDArray
from typing import Callable, Tuple
from data_utils.dataset_generator import generate_static_dataset, generate_cipher_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from neural_network.networks import create_networks
from key.EllipticCurve import generate_key_pair, set_curve, get_key_shape
from argparse import ArgumentParser
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf

#  create method for bitwise add operation with support of multiple dimension arrays a and b
def bitwise_add_carry_multi(a, b):
    """
    a, b: Multi-dimensional NumPy arrays of bits.
    Return: Multi-dimensional NumPy array of bits with the final carry if necessary.
    """
    # Ensure both arrays are the same shape (if not, pad the smaller array)
    print("Applying addition on::")
    print("A")
    print(a.shape)
    print("Whole A")
    print(a)
    print("B")
    print(b.shape)
    print("Whole B\n")

    if a.shape != b.shape:
        max_shape = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))
        a = np.pad(a, [(0, ms - s) for s, ms in zip(a.shape, max_shape)], 'constant', constant_values=0)
        b = np.pad(b, [(0, ms - s) for s, ms in zip(b.shape, max_shape)], 'constant', constant_values=0)

    # If we're at the base case of 1D arrays, use the original bitwise_add_carry function
    if a.ndim == 1:
        return bitwise_add_carry(a, b)

    # Otherwise, recurse over the remaining dimensions
    result = np.zeros_like(a)
    for index in np.ndindex(a.shape[:-1]):  # Iterate over all sub-arrays except the last dimension
        result[index] = bitwise_add_carry(a[index], b[index])

    return result


def bitwise_multiply_multi(a, b):
    """
    Perform bitwise (binary) multiplication on multi-dimensional NumPy arrays a and b of bits.
    The operation is done element-by-element along the last dimension, assuming all preceding
    dimensions match.
    Returns the multi-dimensional array of the product.
    """
    print("Applying multiply_multi on::")
    print("A")
    print(a.shape)
    print("Whole A")
    print(a)
    print("B")
    print(b.shape)
    print("Whole B\n")
    # Ensure both arrays are the same shape by padding the smaller if necessary
    if a.shape != b.shape:
        max_shape = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))
        a = np.pad(a, [(0, ms - s) for s, ms in zip(a.shape, max_shape)],
                   'constant', constant_values=0)
        b = np.pad(b, [(0, ms - s) for s, ms in zip(b.shape, max_shape)],
                   'constant', constant_values=0)

    # Base case: if 1D, use bitwise_multiply.
    if a.ndim == 1:
        return bitwise_multiply(a, b)

    # Otherwise, recurse over the sub-arrays:
    # We'll create an output array with "object" dtype for the last dimension,
    # since each subarray may have different lengths (especially after multiplication).
    # If you need a uniform shape in the last dimension, you might have to
    # determine the maximum length from all sub-results and pad accordingly.
    result_shape = a.shape  # shape for the "outer" dimensions

    result = np.empty(result_shape, dtype=object)

    for index in np.ndindex(result_shape[:-1]):
        # Multiply along the last dimension
        product_1d = bitwise_multiply(a[index], b[index])
        result[index] = product_1d

    print("Result")
    print(result.shape)
    return result

def bitwise_add_carry(a, b): # Expects single dimension arrays
    """
    a, b: 1D NumPy arrays of bits.
    Return: 1D NumPy array of bits with the final carry if necessary.
    """
    max_len = max(len(a), len(b))
    a = np.pad(a, (max_len - len(a), 0), 'constant', constant_values=0)
    b = np.pad(b, (max_len - len(b), 0), 'constant', constant_values=0)

    carry = 0
    result = []

    for i in range(max_len - 1, -1, -1):
        s = a[i] + b[i] + carry
        bit = s % 2

        carry = s // 2
        result.append(bit)

    if carry == 1:
        result.append(carry)

    result.reverse()
    result_array = np.array(result, dtype=np.int64)

    return result_array[-a.shape[0]:]

def bitwise_multiply(a, b):
    """
    Perform bitwise (binary) multiplication on 1D NumPy arrays a and b of bits.
    Returns the product as a 1D NumPy array of bits.
    """
    if len(a) == 0:
        a = np.array([0], dtype=np.int64)
    if len(b) == 0:
        b = np.array([0], dtype=np.int64)

    max_len = len(a) + len(b)
    result = np.zeros(max_len, dtype=np.int64)

    a_rev = a[::-1]
    b_rev = b[::-1]

    for i, bit_b in enumerate(b_rev):
        if bit_b == 1:
            shifted_a = np.concatenate([np.zeros(i, dtype=np.int64), a_rev])
            shifted_a = np.pad(shifted_a, (0, max_len - len(shifted_a)), mode='constant', constant_values=0)
            partial_sum = bitwise_add_carry(result[::-1], shifted_a[::-1])
            result = partial_sum[::-1]

    result = result[::-1]
    idx = 0
    while idx < len(result)-1 and result[idx] == 0:
        idx += 1

    return result[-a.shape[0]:]

class Training:
    def __init__(self, batch_size: int, p1_bits: int, p2_bits: int, c3_bits: int, nonce_bits: int, curve: str, alice: Model, bob: Model, rate: float):
        """Initializes the Training class for training the HO models, Alice, Bob and Eve.
        
        Args:
            batch_size: Number of samples in the dataset.
            p1_bits: Number of bits in plaintext 1.
            p2_bits: Number of bits in plaintext 2.
            c3_bits: Number of bits in the output ciphertext.
            nonce_bits: Number of bits in the nonce.
            curve: Name of the elliptic curve.
            alice: Alice Model.
            bob: Bob Model.
            rate: Dropout rate.
        """
        self.p1_bits = p1_bits
        self.p2_bits = p2_bits
        self.c3_bits = c3_bits
        self.nonce_bits = nonce_bits
        self.curve = curve
        self.alice = alice
        self.bob = bob
        self.batch_size = batch_size
        self.abelosses, self.boblosses, self.evelosses = [], [], []
        self.test_type = f"ma-rate-{rate}-curve-{self.curve.name}-2"
        self.path = f'weights/weights-{self.test_type}'
        isExist = os.path.exists(self.path)
        if not isExist:
            os.makedirs(self.path)

    def train_HO_model(self, HO_model: Model, task: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], op: NDArray[np.object_], filename: str):
        """Train the HO model.

        Args:
            HO_model: HO Model.
            task: Task function which accepts 2 numpy arrays as arguments and returns a single numpy array as the result.
            operation: Operation tag, numpy array of numpy arrays containing float64 elements.
            filename: HO_model filename to save the weights.
        """
        HO_model.trainable = True

        # Train HO model on operation
        X1_train, X2_train, y_train = generate_static_dataset(task, self.c3_bits, self.batch_size, seed=0)
        X1_test, X2_test, y_test = generate_static_dataset(task, self.c3_bits, self.batch_size, mode="extrapolation", seed=0)

        HO_model.fit([op, X1_train, X2_train], y_train, batch_size=128, epochs=512,
        verbose=2, validation_data=([op, X1_test, X2_test], y_test))

        checkpoint = ModelCheckpoint(f'{self.path}/{filename}', monitor='val_loss',
                                verbose=1, save_weights_only=True, save_best_only=True)
        callbacks = [checkpoint]

        # Train HO model with Alice to do addition on encrypted data
        _, public_arr = generate_key_pair(self.batch_size, self.curve)
        X1_cipher_train, X2_cipher_train, y_cipher_train = generate_cipher_dataset(self.p1_bits, self.p2_bits, self.batch_size, public_arr, self.alice, task, self.nonce_bits, 0)
        X1_cipher_test, X2_cipher_test, y_cipher_test = generate_cipher_dataset(self.p1_bits, self.p2_bits, self.batch_size, public_arr, self.alice, task, self.nonce_bits, 1)

        HO_model.fit([op, X1_cipher_train, X2_cipher_train], y_cipher_train, batch_size=128, epochs=512,
            verbose=2, callbacks=callbacks, validation_data=([op, X1_cipher_test, X2_cipher_test], y_cipher_test))

        # Save weights
        HO_model.trainable = False
    
    def generate_batches(self) -> Tuple[NDArray[np.object_], NDArray[np.object_], NDArray[np.object_], NDArray[np.object_], NDArray[np.object_], NDArray[np.object_], NDArray[np.object_]]:
        """Generate batches of data for training.

        Returns:
            p1_batch, p2_batch, private_arr, public_arr, nonce, operation_a and operation_m.
            p1_batch and p2_batch are numpy arrays of numpy arrays containing int64 elements, 
            while private_arr, public_arr, nonce, operation_a and operation_m are numpy arrays of numpy arrays containing float64 elements.
        """
        p1_batch = np.random.randint(
            0, 2, self.p1_bits * self.batch_size).reshape(self.batch_size, self.p1_bits)
        p2_batch = np.random.randint(
            0, 2, self.p2_bits * self.batch_size).reshape(self.batch_size, self.p2_bits)

        private_arr, public_arr = generate_key_pair(self.batch_size, self.curve)

        nonce = np.random.rand(self.batch_size, self.nonce_bits)

        operation_a = np.zeros((self.batch_size, self.c3_bits))
        operation_m = np.ones((self.batch_size, self.c3_bits))

        return p1_batch, p2_batch, private_arr, public_arr, nonce, operation_a, operation_m
    
    def calculate_bob_loss(self, m_enc: NDArray[np.object_], private_arr: NDArray[np.object_], nonce: NDArray[np.object_], expected_output: NDArray[np.object_], operation: str = "") -> np.float64:
        """Calculate the loss for Bob's decryption.

        Args:
            m_enc: Encrypted message.
            private_arr: Private key.
            nonce: Nonce.
            expected_output: Expected output.
            m_enc, private_arr, nonce and expected_output are numpy arrays of numpy arrays containing float64 elements.
        
        Returns:
            The mean of the sum of the absolute differences between the expected output and the decrypted message.
        """
        if operation:
            print(f"\nOperation: {operation}")

        m_dec = self.bob.predict([m_enc, private_arr, nonce])

        print(f"\nm_dec shape: {m_dec.shape}, expected_output shape: {expected_output.shape}")
        print("m_dec0 TYPE")
        print(type(m_dec[0]))
        print("expected_output0 TYPE")
        print(type(expected_output[-1]))
        print(f"m_dec[0] shape: {m_dec[0].shape}, expected_output[0] shape: {expected_output[0].shape}")
        print("\nm_dec[0]")
        print(m_dec[0])
        print("\nexpected_output[0]")
        print(expected_output[0])
        return np.mean(np.sum(np.abs(expected_output - m_dec), axis=-1))
    
    def train(self, HO_model_addition: Model, HO_model_multiplication: Model, eve: Model, abhemodel: Model, evemodel: Model, n_epochs: int, m_train: int):
        """Train on encryption, decryption and eavesdropping.
        
        Args:
            HO_model_addition: HO Addition Model.
            HO_model_multiplication: HO Multiplication Model.
            eve: Eve Model.
            abhemodel: ABHE Model.
            evemodel: Eve Model.
            n_epochs: Number of epochs.
            m_train: Size of the message space.
        """
        alice_weights_path = f'{self.path}/alice_weights.h5'
        bob_weights_path = f'{self.path}/bob_weights.h5'
        eve_weights_path = f'{self.path}/eve_weights.h5'
        n_batches = m_train // self.batch_size # iterations per epoch, training examples divided by batch size
        epoch = 0
        best_abeloss = float('inf')
        best_epoch = 0
        patience_epochs = 20
        while epoch < n_epochs:
            evelosses0 = []
            boblosses0 = []
            abelosses0 = []
            for iteration in range(n_batches):

                # Train the A-B+E network, train both Alice and Bob
                self.alice.trainable = True
                p1_batch, p2_batch, private_arr, public_arr, nonce, operation_a, operation_m = self.generate_batches()
                loss = abhemodel.train_on_batch(
                    [public_arr, p1_batch, p2_batch, nonce, private_arr, operation_a, operation_m], None)  # calculate the loss
                    
                # How well Alice's encryption and Bob's decryption work together
                abelosses0.append(loss)
                self.abelosses.append(loss)
                abeavg = np.mean(abelosses0)

                # Evaluate Bob's ability to decrypt a message
                m1_enc, m2_enc = self.alice.predict([public_arr, p1_batch, p2_batch, nonce])
                m3_enc_a = HO_model_addition.predict([operation_a, m1_enc, m2_enc])
                m3_enc_m = HO_model_multiplication.predict([operation_m, m1_enc, m2_enc])

                loss_m3_a = self.calculate_bob_loss(m3_enc_a, private_arr, nonce, bitwise_add_carry_multi(p1_batch, p2_batch), "addition") # Apply same binary operation
                loss_m3_m = self.calculate_bob_loss(m3_enc_m, private_arr, nonce, bitwise_multiply_multi(p1_batch, p2_batch), "multiplication")
                loss_m1 = self.calculate_bob_loss(m1_enc, private_arr, nonce, p1_batch)
                loss_m2 = self.calculate_bob_loss(m2_enc, private_arr, nonce, p2_batch)
                loss = (loss_m3_a + loss_m3_m + loss_m1 + loss_m2) / 4

                boblosses0.append(loss)
                self.boblosses.append(loss)
                bobavg = np.mean(boblosses0)

                # Train the EVE network
                self.alice.trainable = False
                p1_batch, p2_batch, _, public_arr, nonce, operation_a, operation_m = self.generate_batches()
                loss = evemodel.train_on_batch([public_arr, p1_batch, p2_batch, nonce, operation_a, operation_m], None)

                evelosses0.append(loss)
                self.evelosses.append(loss)
                eveavg = np.mean(evelosses0)

                # Print progress
                if iteration % max(1, (n_batches // 100)) == 0:
                    print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                        epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
                    sys.stdout.flush()

            # Save weights for each improvement in Bob's loss
            epoch_abeloss = np.mean(boblosses0)
            if epoch_abeloss < best_abeloss:
                best_abeloss = epoch_abeloss
                best_epoch = epoch
                self.alice.save_weights(alice_weights_path)
                self.bob.save_weights(bob_weights_path)
                eve.save_weights(eve_weights_path)
                print(f"\nNew best Bob loss {best_abeloss} at epoch {epoch}")
            
            # Early stopping
            if epoch - best_epoch > patience_epochs:
                print(f"\nEarly stopping: No improvement after {patience_epochs} epochs since epoch {best_epoch}. Best Bob loss: {best_abeloss}")
                break

            epoch += 1

        if not os.path.exists(alice_weights_path):
            self.alice.save_weights(alice_weights_path)
            self.bob.save_weights(bob_weights_path)
            eve.save_weights(eve_weights_path)

        print("Training complete.")

    def save_loss_values(self):
        """Save the loss values to a CSV file."""
        steps = -1
        Biodata = {'ABloss': self.abelosses[:steps],
                'Bobloss': self.boblosses[:steps],
                'Eveloss': self.evelosses[:steps]}

        df = pd.DataFrame(Biodata)

        df.to_csv(f'dataset/{self.test_type}.csv', mode='a', index=False)


if __name__ == "__main__":
    # Set the seed for TensorFlow and any other random operation
    seed = 0
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = ArgumentParser()
    parser.add_argument('-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('-epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('-batch', type=int, default=448, help='Batch size')
    parser.add_argument('-curve', type=str, default="secp224r1", help='Elliptic curve name')
    args = parser.parse_args()

    curve = set_curve(args.curve)

    public_bits = get_key_shape(curve)[1]  
    private_bits = get_key_shape(curve)[0]
    dropout_rate = args.rate

    alice, bob, HO_model_addition, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, learning_rate, c3_bits, nonce_bits, HO_model_multiplication = create_networks(public_bits, private_bits, dropout_rate)

    training = Training(args.batch, p1_bits, p2_bits, c3_bits, nonce_bits, curve, alice, bob, args.rate)
    # Suspected culprit for bit addition error
    # Current version has a problem with bit-wise operation.
    # [0,0,0,1] + [0,0,1,1] = [0,0,1,2] (X) - That is x + y = [0,0,1,2]
    # [0,0,0,1] + [0,0,1,1] = [0,1,0,0] (O) - That is, SHOULD BE -> x + y = [0,1,0,0]
    # Create on bit-wise operation function for input. Input is an array
    # Compare plaintext 1 and plaintext 2, and return the result of decrypted result of the addition operation.
    training.train_HO_model(
        HO_model_addition,
        bitwise_add_carry,
        np.zeros((args.batch, c3_bits)),
        "addition_weights.h5"
    )
    training.train_HO_model(HO_model_multiplication,
                            bitwise_multiply,
                            np.ones((args.batch, c3_bits)),
                            "multiplication_weights.h5")
    training.train(HO_model_addition, HO_model_multiplication, eve, abhemodel, evemodel, args.epoch, m_train)
    training.save_loss_values()