import numpy as np
from argparse import ArgumentParser

def save_generated_plaintext(p1_bits: int, p2_bits: int, batch_size: int):
    """Generates two batches of plaintexts and saves them.
    Used to save plaintexts to calculate the mean and standard deviation of these plaintexts when encrypted multiple times using the same key.

    Args:
        p1_bits: Size of the plaintext 1 in bits.
        p1_bits: Size of the plaintext 2 in bits.
        batch_size: Number of samples in the dataset.
    """
    p1_batch = np.random.randint(
        0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
    p2_batch = np.random.randint(
        0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
    np.save(f"plaintext/p1-{batch_size}.npy", p1_batch)
    np.save(f"plaintext/p2-{batch_size}.npy", p2_batch)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-batch', type=int, default=448, help='Batch size')
    args = parser.parse_args()
    p1_bits = 16
    p2_bits = 16
    save_generated_plaintext(p1_bits, p2_bits, args.batch)
