from key.EllipticCurve import get_key_shape, set_curve, generate_key_pair
from data_utils.dataset_generator import generate_static_dataset, generate_cipher_dataset
import numpy as np
from argparse import ArgumentParser
from neural_network.networks_two import create_networks
from keras.callbacks import ModelCheckpoint

parser = ArgumentParser()
parser.add_argument('-op', type=str, default="Adam", help='Optimizer')
parser.add_argument('-lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('-e', type=int, default=1024, help='Number of epochs') #1500
parser.add_argument('-b', type=int, default=128, help='Batch size') #256
args = parser.parse_args()

task_a = lambda x, y: x + y
task_m = lambda x, y: x * y
batch_size = 512

curve = set_curve("secp224r1")
public_bits = get_key_shape(curve)[1]  
private_bits = get_key_shape(curve)[0]

alice, _, HO_model_addition, _, _, _, p1_bits, _, p2_bits, _, c3_bits, nonce_bits, HO_model_multiplication = create_networks(public_bits, private_bits, 0.1)

HO_model_addition.trainable = True

# Train HO model to do addition
X1_train_a, X2_train_a, y_train_a = generate_static_dataset(task_a, c3_bits, batch_size, seed=0)
X1_test_a, X2_test_a, y_test_a = generate_static_dataset(task_a, c3_bits, batch_size, mode="extrapolation", seed=0)

HO_model_addition.fit([X1_train_a, X2_train_a], y_train_a, batch_size=128, epochs=512,
    verbose=2, validation_data=([X1_test_a, X2_test_a], y_test_a))

checkpoint = ModelCheckpoint("ad-weights.h5", monitor='val_loss',
                            verbose=1, save_weights_only=True, save_best_only=True)
callbacks = [checkpoint]

# Train HO model with Alice to do addition on encrypted data
_, public_arr = generate_key_pair(batch_size, curve)
X1_cipher_train_a, X2_cipher_train_a, y_cipher_train_a = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_a, nonce_bits, 0)
X1_cipher_test_a, X2_cipher_test_a, y_cipher_test_a = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_a, nonce_bits, 1)

HO_model_addition.fit([X1_cipher_train_a, X2_cipher_train_a], y_cipher_train_a, batch_size=128, epochs=512,
    verbose=2, callbacks=callbacks, validation_data=([X1_cipher_test_a, X2_cipher_test_a], y_cipher_test_a))

# Save weights
HO_model_addition.trainable = False

HO_model_multiplication.trainable = True

X1_train_m, X2_train_m, y_train_m = generate_static_dataset(task_m, c3_bits, batch_size, seed=1)
X1_test_m, X2_test_m, y_test_m = generate_static_dataset(task_m, c3_bits, batch_size, mode="extrapolation", seed=0)

HO_model_multiplication.fit([X1_train_m, X2_train_m], y_train_m, batch_size=128, epochs=512,
    verbose=2, validation_data=([X1_test_m, X2_test_m], y_test_m))

checkpoint = ModelCheckpoint("mu-weights.h5", monitor='val_loss',
                            verbose=1, save_weights_only=True, save_best_only=True)
callbacks = [checkpoint]

# Train HO model with Alice to do mulitplication on encrypted data
X1_cipher_train_m, X2_cipher_train_m, y_cipher_train_m = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_m, nonce_bits, 2)
X1_cipher_test_m, X2_cipher_test_m, y_cipher_test_m = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_m, nonce_bits, 3)

HO_model_multiplication.fit([X1_cipher_train_m, X2_cipher_train_m], y_cipher_train_m, batch_size=128, epochs=512,
    verbose=2, callbacks=callbacks, validation_data=([X1_cipher_test_m, X2_cipher_test_m], y_cipher_test_m))

# Save weights
HO_model_multiplication.trainable = False

predicted_a = HO_model_addition.predict([X1_test_a, X2_test_a])
print(y_test_a)
print(predicted_a)
print()
predicted_m = HO_model_multiplication.predict([X1_test_m, X2_test_m])
print(y_test_m)
print(predicted_m)

tolerance = 1e-4
correct_elements = np.sum(np.abs(y_test_a - predicted_a) <= tolerance)
total_elements = np.prod(predicted_a.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Addition: {accuracy_percentage:.2f}%")

tolerance = 1e-4
correct_elements = np.sum(np.abs(y_test_m - predicted_m) <= tolerance)
total_elements = np.prod(predicted_m.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Multiplication: {accuracy_percentage:.2f}%")

p1_batch = np.random.randint(
    0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
p2_batch = np.random.randint(
    0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
private_arr, public_arr = generate_key_pair(batch_size, curve)

nonce = np.random.rand(batch_size, nonce_bits)

cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])

cipher_add = HO_model_addition.predict([cipher1, cipher2])

operation = np.ones(cipher1.shape)
cipher_mu = HO_model_multiplication.predict([cipher1, cipher2])

print(cipher1+cipher2)
print(cipher_add)

tolerance = 1e-4
correct_elements = np.sum(np.abs(cipher1+cipher2 - cipher_add) <= tolerance)
total_elements = np.prod(cipher_add.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Addition: {accuracy_percentage:.2f}%")

print(cipher1*cipher2)
print(cipher_mu)

tolerance = 1e-4
correct_elements = np.sum(np.abs(cipher1*cipher2 - cipher_mu) <= tolerance)
total_elements = np.prod(cipher_mu.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Multiplication: {accuracy_percentage:.2f}%")
    
