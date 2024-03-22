import numpy as np
from typing import Callable, Tuple
from tensorflow.keras.models import Model

# Make index selection deterministic as well
np.random.seed(0)

# Matrix containing random shuffled numbers between 0 and 99
# Used to select parts of the input data 
static_index = np.arange(0, 2, dtype=np.int64)
np.random.shuffle(static_index)

# Generates a static dataset based on an operation function 
def generate_static_dataset(
        op_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], 
        num_samples: int = 16, 
        batch_size: int = 512, 
        seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a dataset given an operation. Used to generate static dataset for training.

    Args:
        op_fn: A function which accepts 2 numpy arrays as arguments and returns a single numpy array as the result.
        num_samples: Number of samples for the dataset.
        batch_size: Number of samples in the dataset.
        seed: random seed

    Returns: Dataset x1, x2 and y, where y is the result of the operation on x1 and x2.
    """
    assert callable(op_fn)

    np.random.seed(seed)

    X1_dataset, X2_dataset, y_dataset = [], [], []

    for i in range(batch_size):
        X = np.random.uniform(low=0.0, high=1.00000001, size=(2, num_samples))
        a, b = X  # Unpack the two arrays
        Y = op_fn(a, b)  # Apply the operation

        X1_dataset.append(a)
        X2_dataset.append(b)
        y_dataset.append(Y)

    return np.array(X1_dataset), np.array(X2_dataset), np.array(y_dataset)


def generate_cipher_dataset(
    p1_bits: int, 
    p2_bits: int, 
    batch_size: int, 
    public_arr: np.ndarray, 
    alice: Model, 
    task_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], 
    nonce_bits: int
)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a dataset with ciphertext given an operation. Used to generate a dataset of plaintexts encrypted by Alice.

    Args:
        p1_bits: Number of bits in plaintext 1.
        p2_bits: Number of bits in plaintext 2. 
        batch_size: Number of samples in the dataset. 
        public_arr: Public key. 
        alice: Alice Model.
        task_fn: A function which accepts 2 numpy arrays as arguments and returns a single numpy array as the result., 
        nonce_bits: Number of bits in nonce.

    Returns: Dataset cipher1, cipher2 and cipher3, where cipher3 is the result of the operation on cipher1 and cipher2.
    
    """
    nonce = np.random.rand(batch_size, nonce_bits)
    p1_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
    p2_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
    cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])

    cipher3 = []
    assert callable(task_fn)
    for i in range(len(cipher1)):
        Y = task_fn(cipher1[i], cipher2[i])
        cipher3.append(Y)

    cipher3 = np.array(cipher3)
    return cipher1, cipher2, cipher3

if __name__ == "__main__":
    x1, x2, y = generate_static_dataset(lambda x, y: x + y, 4, 4)
    print(x1)
    print(x2)
    print(y)