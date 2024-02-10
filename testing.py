import numpy as np

# batch_size = 1000
batch_size = 1000
p1_bits = 8
p2_bits = 8

p1_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
p2_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)

X_train = []
y_train = []
for i in range(len(p1_batch)):
    p1_binary_string = ''.join(str(bit) for bit in p1_batch[i])
    p2_binary_string = ''.join(str(bit) for bit in p2_batch[i])
    
    X_train.append([ int(p1_binary_string, 2), int(p2_binary_string, 2)])
    y_train.append([ int(p1_binary_string, 2) + int(p2_binary_string, 2)])

print(X_train)
print(y_train)

