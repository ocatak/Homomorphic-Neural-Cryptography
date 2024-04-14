import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from neural_network.networks_two_meta import create_networks
from key.EllipticCurve import generate_key_pair, set_curve, get_key_shape
from data_utils.dataset_generator import generate_static_dataset, generate_cipher_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import tensorflow as tf
import random

seed = 42
random.seed(seed)
np.random.seed(seed)

parser = ArgumentParser()
parser.add_argument('-rate', type=float, default=0.1, help='Dropout rate')
parser.add_argument('-epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('-batch', type=int, default=512, help='Batch size')
parser.add_argument('-curve', type=str, default="secp224r1", help='Elliptic curve name')
args = parser.parse_args()

curve = set_curve(args.curve)

public_bits = get_key_shape(curve)[1]  
private_bits = get_key_shape(curve)[0]
dropout_rate = args.rate

alice, bob, HO_model_addition, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, learning_rate, c3_bits, nonce_bits, HO_model_multiplication = create_networks(public_bits, private_bits, dropout_rate)

# used to save the results to a different file
test_type = f"ma-rate-{args.rate}-cuvre-{args.curve}-batch-{args.batch}-0.00005-random-52"
optimizer = "Adam"
activation = "tanh-hard-sigmoid-lambda"

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

# Save the results to a text file
with open(f'results/results-{test_type}.txt', "a") as f:
    f.write("Training complete.\n")
    f.write(f"learning rate {learning_rate}\n")
    f.write(f"Optimizer: {optimizer}\n")
    f.write(f"Activation: {activation}\n")
    f.write(f"Curve: {curve.name}")
    f.write(f"Dropout rate: {dropout_rate}")
    f.write("Epochs: {}\n".format(n_epochs))
    f.write("Batch size: {}\n".format(batch_size))
    f.write("Iterations per epoch: {}\n".format(n_batches))
    f.write("Alice-Bob cycles per iteration: {}\n".format(abecycles))
    f.write("Eve cycles per iteration: {}\n".format(evecycles))

    p1_batch = np.random.randint(
        0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
    p2_batch = np.random.randint(
        0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
    private_arr, public_arr = generate_key_pair(batch_size, curve)

    nonce = np.random.rand(batch_size, nonce_bits)

    cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])

    operation_a = np.zeros(cipher1.shape)
    operation_m = np.ones(cipher1.shape)

    cipher_add = HO_model_addition.predict([operation_a, cipher1, cipher2])
    cipher_mu = HO_model_multiplication.predict([operation_m, cipher1, cipher2])

    print("cipher1 + cipher2")
    print(cipher1+cipher2)
    print("HO addition:")
    print(cipher_add)
    print("cipher1 * cipher2")
    print(cipher1*cipher2)
    print("HO multiplication")
    print(cipher_mu)

    tolerance = 1e-3
    correct_elements = np.sum(np.abs(cipher1+cipher2 - cipher_add) <= tolerance)
    total_elements = np.prod(cipher_add.shape)
    accuracy_percentage = (correct_elements / total_elements) * 100
    print(f"HO model Accuracy Percentage Addition: {accuracy_percentage:.2f}%")
    f.write(f"Decryption accuracy by HO addition: {accuracy_percentage:.2f}%\n")

    tolerance = 1e-3
    correct_elements = np.sum(np.abs(cipher1*cipher2 - cipher_mu) <= tolerance)
    total_elements = np.prod(cipher_mu.shape)
    accuracy_percentage = (correct_elements / total_elements) * 100
    print(f"HO model Accuracy Percentage Multiplication: {accuracy_percentage:.2f}%")
    f.write(f"Decryption accuracy by HO multiplication: {accuracy_percentage:.2f}%\n")


    # Bob attempt to decrypt addition
    decrypted = bob.predict([cipher_add, private_arr, nonce])
    decrypted_bits = np.round(decrypted).astype(int)

    print(f"Bob decrypted addition: {decrypted}")
    print(f"Bob decrypted bits addition: {decrypted_bits}")

    # Calculate Bob's decryption accuracy
    correct_bits = np.sum(decrypted_bits == (p1_batch+p2_batch))
    total_bits = np.prod(decrypted_bits.shape)
    accuracy = correct_bits / total_bits * 100

    print(f"Number of correctly decrypted bits addition: {correct_bits}")
    print(f"Total number of bits addition: {total_bits}")
    print(f"Decryption accuracy addition: {accuracy}%")
    f.write(f"Decryption accuracy by Bob Addition: {accuracy}%\n")


    # Bob attempt to decrypt multiplication
    decrypted = bob.predict([cipher_mu, private_arr, nonce])
    decrypted_bits = np.round(decrypted).astype(int)

    print(f"Bob decrypted multiplication: {decrypted}")
    print(f"Bob decrypted bits multiplication: {decrypted_bits}")

    # Calculate Bob's decryption accuracy
    correct_bits = np.sum(decrypted_bits == (p1_batch*p2_batch))
    total_bits = np.prod(decrypted_bits.shape)
    accuracy = correct_bits / total_bits * 100

    print(f"Number of correctly decrypted bits multiplication: {correct_bits}")
    print(f"Total number of bits multiplication: {total_bits}")
    print(f"Decryption accuracy multiplication: {accuracy}%")
    f.write(f"Decryption accuracy by Bob Multiplication: {accuracy}%\n")


    # Eve attempt to decrypt addition
    eve_decrypted = eve.predict([cipher_add, public_arr, nonce])
    eve_decrypted_bits = np.round(eve_decrypted).astype(int)

    print(f"Eve decrypted addition: {eve_decrypted}")
    print(f"Eve decrypted bits addition: {eve_decrypted_bits}")
    
    # Calculate Eve's decryption accuracy
    correct_bits_eve = np.sum(eve_decrypted_bits == (p1_batch+p2_batch))
    total_bits = np.prod(eve_decrypted_bits.shape)
    accuracy_eve = correct_bits_eve / total_bits * 100

    print(f"Number of correctly decrypted bits by Eve addition: {correct_bits_eve}")
    print(f"Total number of bits addition: {total_bits}")
    print(f"Decryption accuracy by Eve addition: {accuracy_eve}%")
    f.write(f"Decryption accuracy by Eve Addition: {accuracy_eve}%\n")


    # Eve attempt to decrypt mulitplication
    eve_decrypted = eve.predict([cipher_mu, public_arr, nonce])
    eve_decrypted_bits = np.round(eve_decrypted).astype(int)

    print(f"Eve decrypted mulitplication: {eve_decrypted}")
    print(f"Eve decrypted bits mulitplication: {eve_decrypted_bits}")
    
    # Calculate Eve's decryption accuracy
    correct_bits_eve = np.sum(eve_decrypted_bits == (p1_batch*p2_batch))
    total_bits = np.prod(eve_decrypted_bits.shape)
    accuracy_eve = correct_bits_eve / total_bits * 100

    print(f"Number of correctly decrypted bits by Eve mulitplication: {correct_bits_eve}")
    print(f"Total number of bits mulitplication: {total_bits}")
    print(f"Decryption accuracy by Eve mulitplication: {accuracy_eve}%")
    f.write(f"Decryption accuracy by Eve Multiplication: {accuracy_eve}%\n")


    # Bob attempt to decrypt cipher1
    decrypted_c1 = bob.predict([cipher1, private_arr, nonce])
    decrypted_bits_c1 = np.round(decrypted_c1).astype(int)

    print(f"Bob decrypted P1: {decrypted_c1}")
    print(f"Bob decrypted bits P1: {decrypted_bits_c1}")

    # Calculate Bob's decryption accuracy
    correct_bits_p1 = np.sum(decrypted_bits_c1 == (p1_batch))
    total_bits_p1 = np.prod(decrypted_bits_c1.shape)
    accuracy_p1 = correct_bits_p1 / total_bits_p1 * 100

    print(f"Number of correctly decrypted bits P1: {correct_bits_p1}")
    print(f"Total number of bits P1: {total_bits_p1}")
    print(f"Decryption accuracy P1: {accuracy_p1}%")

    # Bob attempt to decrypt cipher2
    decrypted_c2 = bob.predict([cipher2, private_arr, nonce])
    decrypted_bits_c2 = np.round(decrypted_c2).astype(int)

    print(f"Bob decrypted P2: {decrypted_c2}")
    print(f"Bob decrypted bits P2: {decrypted_bits_c2}")

    # Calculate Bob's decryption accuracy
    correct_bits_p2 = np.sum(decrypted_bits_c2 == (p2_batch))
    total_bits_p2 = np.prod(decrypted_bits_c1.shape)
    accuracy_p2 = correct_bits_p2 / total_bits_p2 * 100

    print(f"Number of correctly decrypted bits P2: {correct_bits_p2}")
    print(f"Total number of bits P2: {total_bits_p2}")
    print(f"Decryption accuracy P2: {accuracy_p2}%")
