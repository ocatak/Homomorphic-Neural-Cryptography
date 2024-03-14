import numpy as np
from key.EllipticCurve import generate_key_pair, curve

# Make index selection deterministic as well
np.random.seed(0)

# Matrix containing random shuffled numbers between 0 and 99
# Used to select parts of the input data 
static_index = np.arange(0, 2, dtype=np.int64)
np.random.shuffle(static_index)

# Generates a static dataset based on an operation function 
def generate_static_dataset(op_m, op_a, num_samples=572, batch_size=5,seed=0):
    """
    Generates a dataset given an operation.
    Used to generate the synthetic static dataset.

    # Arguments:
        op_fn: A function which accepts 2 numpy arrays as arguments
            and returns a single numpy array as the result.
        num_samples: Number of samples for the dataset.
        batch_size:
        seed: random seed

    Returns:

    """
    assert callable(op_m)
    assert callable(op_a)


    np.random.seed(seed)  # make deterministic

    print("Generating dataset")

    X1_dataset = []
    X2_dataset = []

    y_dataset = []

    for i in range(batch_size):
        # Get the input stream
        X = np.random.uniform(low=0.0, high=1.00000001, size=(num_samples, num_samples))

        a=X[0]
        b=X[1]
        
        c = op_m(a, b)
        Y = op_a(a, c)

        X1_dataset.append(a)
        X2_dataset.append(b)
        y_dataset.append(Y)

    return  np.array(X1_dataset), np.array(X2_dataset), np.array(y_dataset)


def generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, nonce_bits, task_m, task_a):
    nonce = np.random.rand(batch_size, nonce_bits)
    p1_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
    p2_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
    cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])

    cipher3 = []
    assert callable(task_m)
    assert callable(task_a)
    for i in range(len(cipher1)):
        cipher_m = task_m(cipher1[i], cipher2[i])
        Y = task_a(cipher1[i], cipher_m)
        cipher3.append(Y)

    cipher3 = np.array(cipher3)
    return cipher1, cipher2, cipher3

if __name__ == "__main__":
    x1, x2, y = generate_static_dataset(lambda x, y: x * y, lambda x, y: x + y, 2)
    print(x1)
    print(x2)
    print(y)
