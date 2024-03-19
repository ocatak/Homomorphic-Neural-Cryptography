import numpy as np
from key.EllipticCurve import set_curve
from networks_combined import create_networks

# Make index selection deterministic as well
np.random.seed(0)

# Matrix containing random shuffled numbers between 0 and 99
# Used to select parts of the input data 
static_index = np.arange(0, 2, dtype=np.int64)
np.random.shuffle(static_index)

# Generates a static dataset based on an operation function 
def generate_static_dataset(op1, op2, num_samples=572, batch_size=5, seed=0):
    """
    Generates a dataset given an operation.
    Used to generate the synthetic static dataset.

    # Arguments:
        op1: A function which accepts 2 numpy arrays as arguments
            and returns a single numpy array as the result.
        op2: A function which accepts 2 numpy arrays as arguments
            and returns a single numpy array as the result.
        num_samples: Number of samples for the dataset.
        batch_size: Number of samples in the dataset.
        seed: Random seed for reproducibility.

    Returns: Dataset x1, x2, y, where y is the result of op1 and op2 on x1 and x2

    """
    assert callable(op1)
    assert callable(op2)


    np.random.seed(seed)  # make deterministic


    X1_dataset = []
    X2_dataset = []

    y_dataset = []

    for i in range(batch_size):
        # Get the input stream
        X = np.random.uniform(low=0.0, high=1.00000001, size=(num_samples, num_samples))

        a=X[0]
        b=X[1]

        c = op1(a, b)
        Y = op2(a, c)

        X1_dataset.append(a)
        X2_dataset.append(b)
        y_dataset.append(Y)

    return  np.array(X1_dataset), np.array(X2_dataset), np.array(y_dataset)


def generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, nonce_bits, op1, op2):
    """
    Generates two datasets given two operations.
    Used to generate datasets in ciphertext.

    # Arguments:
        p1_bits: Number of bits in plaintext 1.
        p2_bits: Number of bits in plaintext 2.
        batch_size: Number of samples in the dataset.
        public_arr: Public key array.
        alice: Alice model.
        nonce_bits: Number of bits in nonce.
        op1: A function which accepts 2 numpy arrays as arguments and returns a single numpy array as the result.
                First operation performed on the ciphertext.
        op2: A function which accepts 2 numpy arrays as arguments and returns a single numpy array as the result.
                Second operation performed on the ciphertext.
       
    Returns: Dataset x1, x2, y, where y is the result of op1 and op2 on x1 and x2

    """
    nonce = np.random.rand(batch_size, nonce_bits)
    p1_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
    p2_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
    cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])

    # Alice weights are at initialization, so dropout layer will give 0.5
    # Replace 0.5 with 0 to make HO model train on accurate data
    cipher1[cipher1 == 0.5] = 0
    cipher2[cipher2 == 0.5] = 0

    cipher3 = []
    assert callable(op1)
    assert callable(op2)
    for i in range(len(cipher1)):
        cipher_m = op1(cipher1[i], cipher2[i])
        Y = op2(cipher1[i], cipher_m)
        cipher3.append(Y)

    cipher3 = np.array(cipher3)
    return cipher1, cipher2, cipher3

if __name__ == "__main__":
    x1, x2, y = generate_static_dataset(lambda x, y: x * y, lambda x, y: x + y, 512, 512)
    print(x1)
    print(x2)
    print(y)

    rate = 0.1
    curve = set_curve("secp256k1")
    public_arr = np.load(f"key/public_key-{curve.name}.npy")
    private_arr = np.load(f"key/private_key-{curve.name}.npy")
    alice, bob, HO_model, eve, _, _, _, _, _, _, _, nonce_bits = create_networks(public_arr.shape[1], private_arr.shape[1], rate)
    x1, x2, y = generate_cipher_dataset(16, 16, 512, public_arr, alice, nonce_bits, lambda x, y: x * y, lambda x, y: x + y)
