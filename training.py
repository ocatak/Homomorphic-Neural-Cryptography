import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from networks import alice, bob, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, HO_model
from EllipticCurve import generate_key_pair, curve

i = 5 # used to save the results to a different file
curve = curve.name

evelosses = []
boblosses = []
abelosses = []

n_epochs = 2 # number of training epochs
batch_size = 5  # number of training examples utilized in one iteration
n_batches = m_train // batch_size # iterations per epoch, training examples divided by batch size
abecycles = 1  # number of times Alice and Bob network train per iteration
evecycles = 1  # number of times Eve network train per iteration, use 1 or 2.

epoch = 0
start = time.time()
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


            loss = abhemodel.train_on_batch(
                [public_arr, p1_batch, p2_batch, private_arr], None)  # calculate the loss

        # How well Alice's encryption and Bob's decryption work together
        abelosses0.append(loss)
        abelosses.append(loss)
        abeavg = np.mean(abelosses0)

        # Evaluate Bob's ability to decrypt a message
        m1_enc, m2_enc = alice.predict([public_arr, p1_batch, p2_batch])
        m3_enc = HO_model.predict([m1_enc, m2_enc])
        m_dec = bob.predict([m3_enc, private_arr])
        loss = np.mean(np.sum(np.abs(p1_batch + p2_batch - m_dec), axis=-1))
        boblosses0.append(loss)
        boblosses.append(loss)
        bobavg = np.mean(boblosses0)

        #! Eve will not decrypt p_batch, but c3, outcome of HE
        # Train the EVE network
        alice.trainable = False
        for cycle in range(evecycles):
            p1_batch = np.random.randint(
                0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
            p2_batch = np.random.randint(
                0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
            _, public_arr = generate_key_pair(batch_size)
            loss = evemodel.train_on_batch([public_arr, p1_batch, p2_batch], None)
        evelosses0.append(loss)
        evelosses.append(loss)
        eveavg = np.mean(evelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    epoch += 1

print("Training complete.")
end = time.time()
print(end - start)
steps = -1

# Save the loss values to a CSV file
Biodata = {'ABloss': abelosses[:steps],
           'Bobloss': boblosses[:steps],
           'Eveloss': evelosses[:steps]}

df = pd.DataFrame(Biodata)

df.to_csv(f'{curve}/{evecycles}cycle/test-{i}.csv', mode='a', index=False)

plt.figure(figsize=(7, 4))
plt.plot(abelosses[:steps], label='A-B')
plt.plot(evelosses[:steps], label='Eve')
plt.plot(boblosses[:steps], label='Bob')
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)

# save the figure for the loss
plt.savefig(
    f'{curve}/{evecycles}cycle/figures/restult-{i}.png')

# Save the results to a text file
with open('results.txt', "a") as f:
    f.write("Training complete.\n")
    f.write(f"Curve: {curve}")
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

    # Alice encrypts the message
    cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch])

    # HO adds the messages
    cipher3 = HO_model.predict([cipher1, cipher2])

    # Bob attempt to decrypt
    decrypted = bob.predict([cipher3, private_arr])
    decrypted_bits = np.round(decrypted).astype(int)

    # Calculate Bob's decryption accuracy
    correct_bits = np.sum(decrypted_bits == (p1_batch+p2_batch))
    total_bits = np.prod(decrypted_bits.shape)
    accuracy = correct_bits / total_bits * 100

    print(f"Number of correctly decrypted bits: {correct_bits}")
    print(f"Total number of bits: {total_bits}")
    print(f"Decryption accuracy: {accuracy}%")

    # Eve attempt to decrypt
    eve_decrypted = eve.predict(cipher3)
    eve_decrypted_bits = np.round(eve_decrypted).astype(int)
    
    # Calculate Eve's decryption accuracy
    correct_bits_eve = np.sum(eve_decrypted_bits == (p1_batch+p2_batch))
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
