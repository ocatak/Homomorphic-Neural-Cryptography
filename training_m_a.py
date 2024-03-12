import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from networks_combined import create_networks
from key.EllipticCurve import generate_key_pair, curve, get_key_shape
from data_utils_combined import generate_static_dataset, generate_cipher_dataset
from tensorflow.keras.callbacks import ModelCheckpoint

public_bits = get_key_shape()[1]  
private_bits = get_key_shape()[0]
dropout_rate = 0.4

alice, bob, HO_model, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, learning_rate, c3_bits, nonce_bits = create_networks(public_bits, private_bits, dropout_rate)

# used to save the results to a different file
test_type = f"multiplication-rate-{dropout_rate}-curve-{curve.name}"
optimizer = "Adam"
activation = "tanh-hard-sigmoid-lambda"

evelosses = []
boblosses = []
abelosses = []

n_epochs = 50 # number of training epochs
batch_size = 512  # number of training examples utilized in one iteration
n_batches = m_train // batch_size # iterations per epoch, training examples divided by batch size
abecycles = 1  # number of times Alice and Bob network train per iteration
evecycles = 1  # number of times Eve network train per iteration, use 1 or 2.
task_name = 'multiplication-addition'
task_m = lambda x, y: x * y
task_a = lambda x, y: x + y
num_samples = c3_bits

epoch = 0

path = f'weights/weights-{test_type}'

HO_weights_path = f'{path}/{task_name}_weights.h5'
alice_weights_path = f'{path}/alice_weights.h5'
bob_weights_path = f'{path}/bob_weights.h5'
eve_weights_path = f'{path}/eve_weights.h5'

isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)

HO_model.trainable = True

# Train HO model to do addition
X1_train, X2_train, y_train = generate_static_dataset(task_m, task_a, num_samples, batch_size, nonce_bits)
X1_test, X2_test, y_test = generate_static_dataset(task_m, task_a, num_samples, batch_size, nonce_bits)

HO_model.fit([X1_train, X2_train], y_train, batch_size=128, epochs=512,
    verbose=2, validation_data=([X1_test, X2_test], y_test))

checkpoint = ModelCheckpoint(HO_weights_path, monitor='val_loss',
                            verbose=1, save_weights_only=True, save_best_only=True)

callbacks = [checkpoint]

private_arr, public_arr = generate_key_pair(batch_size)
# Train HO model with Alice to do addition on encrypted data
X1_cipher_train, X2_cipher_train, y_cipher_train = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, nonce_bits, task_m, task_a)
X1_cipher_test, X2_cipher_test, y_cipher_test = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, nonce_bits, task_m, task_a)

HO_model.fit([X1_cipher_train, X2_cipher_train], y_cipher_train, batch_size=128, epochs=512,
    verbose=2, callbacks=callbacks, validation_data=([X1_cipher_test, X2_cipher_test], y_cipher_test))

# Save weights
HO_model.trainable = False

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

            private_arr, public_arr = generate_key_pair(batch_size)

            nonce = np.random.rand(batch_size, nonce_bits)

            loss = abhemodel.train_on_batch(
                [public_arr, p1_batch, p2_batch, nonce, private_arr], None)  # calculate the loss

        # How well Alice's encryption and Bob's decryption work together
        abelosses0.append(loss)
        abelosses.append(loss)
        abeavg = np.mean(abelosses0)

        # Evaluate Bob's ability to decrypt a message
        m1_enc, m2_enc = alice.predict([public_arr, p1_batch, p2_batch, nonce])
        m3_enc = HO_model.predict([m1_enc, m2_enc])
        m3_dec = bob.predict([m3_enc, private_arr, nonce])
        loss_m3 = np.mean(np.sum(np.abs(p1_batch * p2_batch + p1_batch - m3_dec), axis=-1))

        m1_dec = bob.predict([m1_enc, private_arr, nonce])
        loss_m1 = np.mean(np.sum(np.abs(p1_batch - m1_dec), axis=-1))

        loss = (loss_m3+loss_m1)/2

        boblosses0.append(loss)
        boblosses.append(loss)
        bobavg = np.mean(boblosses0)

        # Train the EVE network
        alice.trainable = False
        for cycle in range(evecycles):
            p1_batch = np.random.randint(
                0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
            p2_batch = np.random.randint(
                0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
            _, public_arr = generate_key_pair(batch_size)
            nonce = np.random.rand(batch_size, nonce_bits)

            loss = evemodel.train_on_batch([public_arr, p1_batch, p2_batch, nonce], None)
        evelosses0.append(loss)
        evelosses.append(loss)
        eveavg = np.mean(evelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    epoch += 1

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
    f.write(f"P_bits {p1_bits}\n")
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

    # Test the model
    p1_batch = np.random.randint(
        0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
    p2_batch = np.random.randint(
        0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
    private_arr, public_arr = generate_key_pair(batch_size)

    nonce = np.random.rand(batch_size, nonce_bits)

    print(f"P1: {p1_batch}")
    print(f"P2: {p2_batch}")

    # Alice encrypts the message
    cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])
    print(f"Cipher1: {cipher1}")
    print(f"Cipher2: {cipher2}")

    # HO adds the messages
    cipher3 = HO_model.predict([cipher1, cipher2])
    print(f"Cipher3: {cipher3}")

    # Bob attempt to decrypt
    decrypted = bob.predict([cipher3, private_arr, nonce])
    decrypted_bits = np.round(decrypted).astype(int)

    print(f"Bob decrypted: {decrypted}")
    print(f"Bob decrypted bits: {decrypted_bits}")

    # Calculate Bob's decryption accuracy
    correct_bits = np.sum(decrypted_bits == (p1_batch * p2_batch + p1_batch))
    total_bits = np.prod(decrypted_bits.shape)
    accuracy = correct_bits / total_bits * 100

    print(f"Number of correctly decrypted bits: {correct_bits}")
    print(f"Total number of bits: {total_bits}")
    print(f"Decryption accuracy: {accuracy}%")

    # Eve attempt to decrypt
    eve_decrypted = eve.predict([cipher3, public_arr, nonce])
    eve_decrypted_bits = np.round(eve_decrypted).astype(int)

    print(f"Eve decrypted: {eve_decrypted}")
    print(f"Eve decrypted bits: {eve_decrypted_bits}")
    
    # Calculate Eve's decryption accuracy
    correct_bits_eve = np.sum(eve_decrypted_bits == (p1_batch * p2_batch + p1_batch))
    total_bits = np.prod(eve_decrypted_bits.shape)
    accuracy_eve = correct_bits_eve / total_bits * 100

    print(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}")
    print(f"Total number of bits: {total_bits}")
    print(f"Decryption accuracy by Eve: {accuracy_eve}%")

    f.write(f"Total number of bits: {total_bits}\n")
    f.write(f"Number of correctly decrypted bits by Bob: {correct_bits}\n")
    f.write(f"Decryption accuracy by Bob: {accuracy}%\n")
    f.write(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}\n")
    f.write(f"Decryption accuracy by Eve: {accuracy_eve}%\n")
    f.write("\n")

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
