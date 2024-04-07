import numpy as np

def generate_balanced_addition_batch(batch_size, p1_bits, p2_bits):
    # Initialize the plaintext arrays
    p1_add = np.zeros((batch_size, p1_bits), dtype=int)
    p2_add = np.zeros((batch_size, p2_bits), dtype=int)
    
    # Ensure that the batch has an equal number of '0', '1', and '2' outcomes
    third_batch_size = batch_size // 3
    for i in range(third_batch_size):
        p1_add[i] = np.random.randint(0, 2, p1_bits)  # Random 0 or 1 for first third
        p2_add[i] = 0  # This will guarantee a '0' or '1' sum
        
    for i in range(third_batch_size, 2*third_batch_size):
        p1_add[i] = np.random.randint(0, 2, p1_bits)  # Random 0 or 1 for second third
        p2_add[i] = 1 - p1_add[i]  # This will guarantee a sum of '1'
        
    for i in range(2*third_batch_size, batch_size):
        p1_add[i] = 1  # Set all bits to 1 for last third
        p2_add[i] = 1  # This will guarantee a sum of '2'
        
    # Shuffle the batches to prevent the network from learning the order
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    p1_add = p1_add[indices]
    p2_add = p2_add[indices]
    
    return p1_add, p2_add


p1, p2 = generate_balanced_addition_batch(6, 4, 4)
print(p1+p2)

p1_add = np.random.randint(
    0, 2, 4 * 6).reshape(6, 4)
p2_add = np.random.randint(
    0, 2, 4 * 6).reshape(6, 4)
print(p1_add+p2_add)