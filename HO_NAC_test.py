from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Lambda, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from key.EllipticCurve import get_key_shape, generate_key_pair, curve
from nac import NAC
from data_utils import generate_static_dataset, generate_cipher_dataset
from networks_functions import create_networks
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

c1_bits = 16
c2_bits = 16
c3_bits = 16
task_fn = lambda x, y: x + y
batch_size = 512

public_bits = get_key_shape()[1]  
private_bits = get_key_shape()[0]

alice, _, HO_model, _, _, _, p1_bits, _, p2_bits, _, c3_bits, nonce_bits = create_networks(public_bits, private_bits, 0)

X1_train, X2_train, y_train = generate_static_dataset(task_fn, c3_bits, batch_size)
X1_test, X2_test, y_test = generate_static_dataset(task_fn, c3_bits, batch_size)

HO_model.fit([X1_train, X2_train], y_train, batch_size=128, epochs=512,
    verbose=2, validation_data=([X1_test, X2_test], y_test))

checkpoint = ModelCheckpoint("test-weight.h5", monitor='val_loss',
                            verbose=1, save_weights_only=True, save_best_only=True)

callbacks = [checkpoint]

private_arr, public_arr = generate_key_pair(batch_size)
# Train HO model with Alice to do addition on encrypted data
X1_cipher_train, X2_cipher_train, y_cipher_train = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_fn, nonce_bits)
X1_cipher_test, X2_cipher_test, y_cipher_test = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_fn, nonce_bits)

HO_model.fit([X1_cipher_train, X2_cipher_train], y_cipher_train, batch_size=128, epochs=512,
    verbose=2, callbacks=callbacks, validation_data=([X1_cipher_test, X2_cipher_test], y_cipher_test))


HO_model.trainable = False

# predicted = HO_model.predict([X1_test, X2_test], 128)
# print(X1_test[:1])
# print(X2_test[:1])
# print(y_test[:1])
# print(predicted[:1])

p1_batch = np.random.randint(
    0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
p2_batch = np.random.randint(
    0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
private_arr, public_arr = generate_key_pair(batch_size)

nonce = np.random.rand(batch_size, nonce_bits)

cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])

predicted = HO_model.predict([cipher1, cipher2])


tolerance = 1e-4
correct_elements = np.sum(np.abs(cipher1+cipher2 - predicted) <= tolerance)
total_elements = np.prod(predicted.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Addition: {accuracy_percentage:.2f}%")