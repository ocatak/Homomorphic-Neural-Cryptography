import numpy as np
p1_bits = 16
p2_bits = 16
batch_size = 512

p1_batch = np.random.randint(
    0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
p2_batch = np.random.randint(
    0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')

np.save("plaintext/p1_batch.npy", p1_batch)
np.save("plaintext/p2_batch.npy", p2_batch)