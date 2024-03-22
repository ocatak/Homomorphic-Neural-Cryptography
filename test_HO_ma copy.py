from key.EllipticCurve import get_key_shape, set_curve, generate_key_pair
from data_utils.dataset_generator import generate_static_dataset, generate_cipher_dataset
import numpy as np
from argparse import ArgumentParser
from neural_network.networks_functions import create_networks
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

alice, _, HO_model, _, _, _, p1_bits, _, p2_bits, _, c3_bits, nonce_bits = create_networks(public_bits, private_bits, 0)


HO_model.trainable = True

X1_train_a, X2_train_a, y_train_a = generate_static_dataset(task_a, c3_bits, batch_size)
X1_test_a, X2_test_a, y_test_a = generate_static_dataset(task_a, c3_bits, batch_size)
op_a = np.zeros(X1_train_a.shape)

X1_train_m, X2_train_m, y_train_m = generate_static_dataset(task_m, c3_bits, batch_size)
X1_test_m, X2_test_m, y_test_m = generate_static_dataset(task_m, c3_bits, batch_size)
op_m = np.ones(X1_train_m.shape)

X1_train = np.concatenate((X1_train_a, X1_train_m))
X2_train = np.concatenate((X2_train_a, X2_train_m))
y_train = np.concatenate((y_train_a, y_train_m))
X1_test = np.concatenate((X1_test_a, X1_test_m))
X2_test = np.concatenate((X2_test_a, X2_test_m))
y_test = np.concatenate((y_test_a, y_test_m))
operation = np.concatenate((op_a, op_m))

HO_model.fit([operation, X1_train, X2_train], y_train, batch_size=args.b, epochs=args.e,
    verbose=2, validation_data=([operation, X1_test, X2_test], y_test))

checkpoint = ModelCheckpoint("HO-test-weights.h5", monitor='val_loss',
                            verbose=1, save_weights_only=True, save_best_only=True)

callbacks = [checkpoint]

private_arr, public_arr = generate_key_pair(batch_size, curve)
# Train HO model with Alice to do addition on encrypted data
X1_cipher_train_a, X2_cipher_train_a, y_cipher_train_a = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_a, nonce_bits)
X1_cipher_test_a, X2_cipher_test_a, y_cipher_test_a = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_a, nonce_bits)
cipher_operation_a = np.zeros(X1_cipher_train_a.shape)

X1_cipher_train_m, X2_cipher_train_m, y_cipher_train_m = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_m, nonce_bits)
X1_cipher_test_m, X2_cipher_test_m, y_cipher_test_m = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_m, nonce_bits)
cipher_operation_m = np.ones(X1_cipher_train_m.shape)

X1_cipher_train = np.concatenate((X1_cipher_train_a, X1_cipher_train_m))
X2_cipher_train = np.concatenate((X2_cipher_train_a, X2_cipher_train_m))
y_cipher_train = np.concatenate((y_cipher_train_a, y_cipher_train_m))
X1_cipher_test = np.concatenate((X1_cipher_test_a, X1_cipher_test_m))
X2_cipher_test = np.concatenate((X2_cipher_test_a, X2_cipher_test_m))
y_cipher_test = np.concatenate((y_cipher_test_a, y_cipher_test_m))
cipher_operation = np.concatenate((cipher_operation_a, cipher_operation_m))


HO_model.fit([cipher_operation, X1_cipher_train, X2_cipher_train], y_cipher_train, batch_size=128, epochs=512,
    verbose=2, callbacks=callbacks, validation_data=([cipher_operation, X1_cipher_test, X2_cipher_test], y_cipher_test))


HO_model.trainable = False


predicted_a = HO_model.predict([op_a, X1_test_a, X2_test_a], 128)
print(X1_test_a[:1])
print(X2_test_a[:1])
print(y_test_a[:1])
print(predicted_a[:1])
print()
predicted_m = HO_model.predict([op_m, X1_test_m, X2_test_m], 128)
print(X1_test_m[:1])
print(X2_test_m[:1])
print(y_test_m[:1])
print(predicted_m[:1])

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

print(f"epoch {args.e}, batch {args.b}, optimizer {args.op}, learning rate {args.lr}")

p1_batch = np.random.randint(
    0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
p2_batch = np.random.randint(
    0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
private_arr, public_arr = generate_key_pair(batch_size, curve)

nonce = np.random.rand(batch_size, nonce_bits)

cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])

operation = np.zeros(cipher1.shape)
cipher_add = HO_model.predict([operation, cipher1, cipher2])

operation = np.ones(cipher1.shape)
cipher_mu = HO_model.predict([operation, cipher1, cipher2])

print(cipher1+cipher2)
print(cipher_add)
print(cipher1*cipher2)
print(cipher_mu)

tolerance = 1e-4
correct_elements = np.sum(np.abs(cipher1+cipher2 - cipher_add) <= tolerance)
total_elements = np.prod(cipher_add.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Addition: {accuracy_percentage:.2f}%")


tolerance = 1e-4
correct_elements = np.sum(np.abs(cipher1*cipher2 - cipher_mu) <= tolerance)
total_elements = np.prod(cipher_mu.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Multiplication: {accuracy_percentage:.2f}%")
    
