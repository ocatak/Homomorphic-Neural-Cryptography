import numpy as np
from typing import Callable, Tuple
from tensorflow.keras.models import Model
from numpy.typing import NDArray

# Make index selection deterministic as well
np.random.seed(0)

# Matrix containing random shuffled numbers between 0 and 99
# Used to select parts of the input data 
static_index = np.arange(0, 2, dtype=np.int64)
np.random.shuffle(static_index)

# Generates a static dataset based on an operation function 
def generate_static_dataset(
        op_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], 
        num_samples: int = 16, 
        batch_size: int = 512, 
        seed: int = 0,
        mode: str = "interpolation"
) -> Tuple[NDArray[np.object_], NDArray[np.object_], NDArray[np.object_]]:
    """Generates a dataset given an operation. Used to generate static dataset for training.

    Args:
        op_fn: A function which accepts 2 numpy arrays as arguments and returns a single numpy array as the result.
        num_samples: Size of the input and output arrays.
        batch_size: Number of samples in the dataset.
        seed: random seed

    Returns: Tuple of (X1, X2, y), where X1, X2 and y are numpy arrays of numpy arrays containing float64 elements. 
        y is the result of the operation on X1 and X2.
    """
    assert callable(op_fn)

    np.random.seed(seed)

    X1_dataset, X2_dataset, y_dataset = [], [], []

    for i in range(batch_size):
        X = np.random.uniform(low=0.0, high=1.00000001, size=(2, num_samples))
        if mode == "extrapolation":
            X *= 0.1
        a, b = X  # Unpack the two arrays

        zero_fraction = 0.2
        num_zeros = int(num_samples * zero_fraction)
        
        # Randomly choose indices to set to zero
        indices_to_zero = np.random.choice(num_samples, num_zeros, replace=False)
        a[indices_to_zero] = 0
        b[indices_to_zero] = 0

        # Apply the operation to the shuffled data
        Y = op_fn(a, b)  # Apply the operation

        # Generating the data set, X1, X2
        # X1 is the first input to the operation
        # X2 is the second input to the operation
        # Y is the output of the operation
        X1_dataset.append(a)
        X2_dataset.append(b)
        y_dataset.append(Y)

    return np.array(X1_dataset), np.array(X2_dataset), np.array(y_dataset)


def generate_cipher_dataset(
    p1_bits: int, 
    p2_bits: int, 
    batch_size: int, 
    public_arr: NDArray[np.object_], 
    alice: Model, 
    task_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], 
    nonce_bits: int,
    seed: int = 0,
)-> Tuple[NDArray[np.object_], NDArray[np.object_], NDArray[np.object_]]:
    """Generates a dataset with ciphertext given an operation. Used to generate a dataset of ciphertexts encrypted by Alice.

    Args:
        p1_bits: Number of bits in plaintext 1.
        p2_bits: Number of bits in plaintext 2. 
        batch_size: Number of samples in the dataset. 
        public_arr: Public key, an numpy array of numpy arrays containing float64 elements. 
        alice: Alice Model.
        task_fn: A function which accepts 2 numpy arrays as arguments and returns a single numpy array as the result. 
        nonce_bits: Number of bits in nonce.

    Returns: Tuple of (cipher1, cipher2, cipher3), where cipher1, cipher2 and cipher3 are numpy arrays of numpy arrays containing float64 elements. 
        cipher3 is the result of the operation on cipher1 and cipher2.
    
    """
    np.random.seed(seed)
    nonce = np.random.rand(batch_size, nonce_bits)
    p1_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
    p2_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
    cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])

    # Alice weights are at initialization, so dropout layer will give 0.5
    # Replace 0.5 with 0 to make HO model train on accurate data
    cipher1[cipher1 == 0.5] = 0
    cipher2[cipher2 == 0.5] = 0

    cipher3 = []
    assert callable(task_fn)
    for i in range(len(cipher1)):
        Y = task_fn(cipher1[i], cipher2[i])
        cipher3.append(Y)

    cipher3 = np.array(cipher3)
    return cipher1, cipher2, cipher3

if __name__ == "__main__":
    x1, x2, y = generate_static_dataset(lambda x, y: x + y, 4, 4)