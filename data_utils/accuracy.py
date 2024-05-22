from tensorflow.keras.models import Model
import numpy as np
from numpy.typing import NDArray

def decryption_accurancy(model: Model, cipher: NDArray[np.object_], key_arr: NDArray[np.object_], nonce: int, p_batch: NDArray[np.object_]) -> np.float64:
    """Calculates the decryption accuracy of the model.
    
    Args:
        model: Model.
        cipher: Ciphertext, a numpy array of numpy arrays consisting of float32 elements.
        key_arr: Key array, a numpy array of numpy arrays consisting of float64 elements.
        nonce: Nonce.
        p_batch: Plaintext batch, a numpy array of numpy arrays consisting of float32 elements.

    Returns:
        The decryption accuracy of the model.
    """
    # Calculate Bob's decryption accuracy
    decrypted = model.predict([cipher, key_arr, nonce])
    decrypted_bits = np.round(decrypted).astype(int)
    correct_bits = np.sum(decrypted_bits == (p_batch))
    total_bits = np.prod(decrypted_bits.shape)
    return np.round(correct_bits / total_bits * 100, 2)

def HO_accuracy(cipher: NDArray[np.object_], computed_cipher: NDArray[np.object_]):
    """Calculates the accuracy of the HO model.

    Args:
        cipher: Ciphertext, a numpy array of numpy arrays consisting of float32 elements.
        computed_cipher: Computed ciphertext, a numpy array of numpy arrays consisting of float32 elements.
    
    Returns:
        The accuracy of the HO model.
    """
    tolerance = 1e-3
    correct_elements = np.sum(np.abs(computed_cipher - cipher) <= tolerance)
    total_elements = np.prod(cipher.shape)
    return (correct_elements / total_elements) * 100
     