import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from neural_network.networks import create_networks
from key.EllipticCurve import generate_key_pair, set_curve, get_key_shape
from data_utils.dataset_generator import generate_static_dataset, generate_cipher_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import random

# Set the seed for TensorFlow and any other random operation
seed = 0
tf.compat.v1.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

parser = ArgumentParser()
parser.add_argument('-rate', type=float, default=0.1, help='Dropout rate')
parser.add_argument('-epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('-batch', type=int, default=448, help='Batch size')
parser.add_argument('-curve', type=str, default="secp224r1", help='Elliptic curve name')
args = parser.parse_args()

curve = set_curve(args.curve)

public_bits = get_key_shape(curve)[1]  
private_bits = get_key_shape(curve)[0]
dropout_rate = args.rate

alice, bob, HO_model_addition, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, learning_rate, c3_bits, nonce_bits, HO_model_multiplication = create_networks(public_bits, private_bits, dropout_rate)

# used to save the results to a different file
test_type = f"ma-rate-{args.rate}-curve-{args.curve}-extra-out"

best_abeloss = float('inf')
best_epoch = 0
patience_epochs = 5

evelosses = []
boblosses = []
abelosses = []

n_epochs = args.epoch # number of training epochs
batch_size = args.batch  # number of training examples utilized in one iteration
n_batches = m_train // args.batch # iterations per epoch, training examples divided by batch size
abecycles = 1  # number of times Alice and Bob network train per iteration
evecycles = 1  # number of times Eve network train per iteration, use 1 or 2.
task_m = lambda x, y: x * y
task_a = lambda x, y: x + y
num_samples = c3_bits

epoch = 0

path = f'weights/weights-{test_type}'

HO_weights_addition_path = f'{path}/addition_weights.h5'
HO_weights_multiplication_path = f'{path}/multiplication_weights.h5'
alice_weights_path = f'{path}/alice_weights.h5'
bob_weights_path = f'{path}/bob_weights.h5'
eve_weights_path = f'{path}/eve_weights.h5'

isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)

HO_model_addition.trainable = True

# Train HO model to do addition
X1_train_a, X2_train_a, y_train_a = generate_static_dataset(task_a, c3_bits, batch_size, seed=0)
X1_test_a, X2_test_a, y_test_a = generate_static_dataset(task_a, c3_bits, batch_size, mode="extrapolation", seed=0)
op_a = np.zeros(X1_train_a.shape)

HO_model_addition.fit([op_a, X1_train_a, X2_train_a], y_train_a, batch_size=128, epochs=512,
    verbose=2, validation_data=([op_a, X1_test_a, X2_test_a], y_test_a))

checkpoint = ModelCheckpoint(HO_weights_addition_path, monitor='val_loss',
                            verbose=1, save_weights_only=True, save_best_only=True)
callbacks = [checkpoint]

# Train HO model with Alice to do addition on encrypted data
_, public_arr = generate_key_pair(batch_size, curve)
X1_cipher_train_a, X2_cipher_train_a, y_cipher_train_a = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_a, nonce_bits, 0)
X1_cipher_test_a, X2_cipher_test_a, y_cipher_test_a = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_a, nonce_bits, 1)
op_a = np.zeros(X1_cipher_train_a.shape)

HO_model_addition.fit([op_a, X1_cipher_train_a, X2_cipher_train_a], y_cipher_train_a, batch_size=128, epochs=512,
    verbose=2, callbacks=callbacks, validation_data=([op_a, X1_cipher_test_a, X2_cipher_test_a], y_cipher_test_a))

# Save weights
HO_model_addition.trainable = False

HO_model_multiplication.trainable = True

X1_train_m, X2_train_m, y_train_m = generate_static_dataset(task_m, c3_bits, batch_size, seed=1)
X1_test_m, X2_test_m, y_test_m = generate_static_dataset(task_m, c3_bits, batch_size, mode="extrapolation", seed=0)
op_m = np.ones(X1_train_m.shape)

HO_model_multiplication.fit([op_m, X1_train_m, X2_train_m], y_train_m, batch_size=128, epochs=512,
    verbose=2, validation_data=([op_m, X1_test_m, X2_test_m], y_test_m))

checkpoint = ModelCheckpoint(HO_weights_multiplication_path, monitor='val_loss',
                            verbose=1, save_weights_only=True, save_best_only=True)
callbacks = [checkpoint]

# Train HO model with Alice to do mulitplication on encrypted data
X1_cipher_train_m, X2_cipher_train_m, y_cipher_train_m = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_m, nonce_bits, 2)
X1_cipher_test_m, X2_cipher_test_m, y_cipher_test_m = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_m, nonce_bits, 3)
op_m = np.ones(X1_cipher_train_m.shape)

HO_model_multiplication.fit([op_m, X1_cipher_train_m, X2_cipher_train_m], y_cipher_train_m, batch_size=128, epochs=512,
    verbose=2, callbacks=callbacks, validation_data=([op_m, X1_cipher_test_m, X2_cipher_test_m], y_cipher_test_m))

# Save weights
HO_model_multiplication.trainable = False

while epoch < n_epochs:
    evelosses0 = []
    boblosses0 = []
    abelosses0 = []
    for iteration in range(n_batches):

        # Train the A-B+E network, train both Alice and Bob
        alice.trainable = True
        for cycle in range(abecycles):
             # Select two random batches of plaintexts
            p1_batch = np.random.randint(
                0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
            p2_batch = np.random.randint(
                0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)

            private_arr, public_arr = generate_key_pair(batch_size, curve)

            nonce = np.random.rand(batch_size, nonce_bits)

            operation_a = np.zeros((batch_size, c3_bits))
            operation_m = np.ones((batch_size, c3_bits))

            loss = abhemodel.train_on_batch(
                [public_arr, p1_batch, p2_batch, nonce, private_arr, operation_a, operation_m], None)  # calculate the loss
            
        # How well Alice's encryption and Bob's decryption work together
        abelosses0.append(loss)
        abelosses.append(loss)
        abeavg = np.mean(abelosses0)

         # Evaluate Bob's ability to decrypt a message
        m1_enc, m2_enc = alice.predict([public_arr, p1_batch, p2_batch, nonce])
        m3_enc_a = HO_model_addition.predict([operation_a, m1_enc, m2_enc])
        m3_enc_m = HO_model_multiplication.predict([operation_m, m1_enc, m2_enc])

        m3_dec_a = bob.predict([m3_enc_a, private_arr, nonce])
        loss_m3_a = np.mean(np.sum(np.abs(p1_batch + p2_batch - m3_dec_a), axis=-1))

        m3_dec_m = bob.predict([m3_enc_m, private_arr, nonce])
        loss_m3_m = np.mean(np.sum(np.abs(p1_batch * p2_batch - m3_dec_m), axis=-1))

        m1_dec = bob.predict([m1_enc, private_arr, nonce])
        loss_m1 = np.mean(np.sum(np.abs(p1_batch - m1_dec), axis=-1))

        m2_dec = bob.predict([m2_enc, private_arr, nonce])
        loss_m2 = np.mean(np.sum(np.abs(p2_batch - m2_dec), axis=-1))

        loss = (loss_m3_a + loss_m3_m + loss_m1 + loss_m2) / 4

        boblosses0.append(loss)
        boblosses.append(loss)
        bobavg = np.mean(boblosses0)

        # Train the EVE network
        alice.trainable = False
        for cycle in range(evecycles):
            # Select two random batches of plaintexts
            p1_batch = np.random.randint(
                0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
            p2_batch = np.random.randint(
                0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)

            private_arr, public_arr = generate_key_pair(batch_size, curve)

            nonce = np.random.rand(batch_size, nonce_bits)

            operation_a = np.zeros((batch_size, c3_bits))
            operation_m = np.ones((batch_size, c3_bits))

            loss = evemodel.train_on_batch([public_arr, p1_batch, p2_batch, nonce, operation_a, operation_m], None)

        evelosses0.append(loss)
        evelosses.append(loss)
        eveavg = np.mean(evelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    epoch_abeloss = np.mean(boblosses0)
    if epoch_abeloss < best_abeloss:
        best_abeloss = epoch_abeloss
        best_epoch = epoch
        alice.save_weights(alice_weights_path)
        bob.save_weights(bob_weights_path)
        eve.save_weights(eve_weights_path)
        print(f"\nNew best Bob loss {best_abeloss} at epoch {epoch}")
    
    if epoch - best_epoch > patience_epochs:
        print(f"\nEarly stopping: No improvement after {patience_epochs} epochs since epoch {best_epoch}. Best Bob loss: {best_abeloss}")
        break

    epoch += 1

if not os.path.exists(alice_weights_path):
    alice.save_weights(alice_weights_path)
    bob.save_weights(bob_weights_path)
    eve.save_weights(eve_weights_path)

print("Training complete.")
steps = -1

# Save the loss values to a CSV file
Biodata = {'ABloss': abelosses[:steps],
           'Bobloss': boblosses[:steps],
           'Eveloss': evelosses[:steps]}

df = pd.DataFrame(Biodata)

df.to_csv(f'dataset/{test_type}.csv', mode='a', index=False)

plt.figure(figsize=(7, 4))
plt.plot(abelosses[:steps], label='A-B')
plt.plot(evelosses[:steps], label='Eve')
plt.plot(boblosses[:steps], label='Bob')
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)

# save the figure for the loss
plt.savefig(
    f'figures/{test_type}.png')
