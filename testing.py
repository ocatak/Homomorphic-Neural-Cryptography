from networks import HO_model, alice, bob, eve, p1_bits, p2_bits
import numpy as np
from EllipticCurve import generate_key_pair

batch_size = 512
HO_weights_path = 'weights-check-input-2/addition_weights.h5'
# alice_weights_path = 'weights-check-input-2/alice_weights.h5'
# bob_weights_path = 'weights-check-input-2/bob_weights.h5'
# eve_weights_path = 'weights-check-input-2/eve_weights.h5'

HO_model.load_weights(HO_weights_path)
# alice.load_weights(alice_weights_path)
# bob.load_weights(bob_weights_path)
# eve.load_weights(eve_weights_path)

p1_batch = np.random.randint(
    0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
p2_batch = np.random.randint(
    0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')
private_arr, public_arr = generate_key_pair(batch_size)

print(f"P1: {p1_batch}")
print(f"P2: {p2_batch}")

# Alice encrypts the message
cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch])
print(f"Cipher1: {cipher1}")
print(f"Cipher2: {cipher2}")

# HO adds the messages
cipher3 = HO_model.predict([cipher1, cipher2])
computed_cipher = cipher1 + cipher2
tolerance = 1e-5
correct_elements = np.sum(np.abs(computed_cipher - cipher3) <= tolerance)
total_elements = np.prod(cipher3.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"Correct Elements: {correct_elements}")
print(f"Total Elements: {total_elements}")
print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")
exit()

# Bob attempt to decrypt
decrypted = bob.predict([cipher3, private_arr])
decrypted_bits = np.round(decrypted).astype(int)

print(f"Bob decrypted: {decrypted}")
print(f"Bob decrypted bits: {decrypted_bits}")

# Calculate Bob's decryption accuracy
correct_bits = np.sum(decrypted_bits == (p1_batch+p2_batch))
total_bits = np.prod(decrypted_bits.shape)
accuracy = correct_bits / total_bits * 100

print(f"Number of correctly decrypted bits: {correct_bits}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy: {accuracy}%")

# Eve attempt to decrypt
eve_decrypted = eve.predict([cipher3, public_arr])
eve_decrypted_bits = np.round(eve_decrypted).astype(int)

print(f"Eve decrypted: {eve_decrypted}")
print(f"Eve decrypted bits: {eve_decrypted_bits}")

# Calculate Eve's decryption accuracy
correct_bits_eve = np.sum(eve_decrypted_bits == (p1_batch+p2_batch))
total_bits = np.prod(eve_decrypted_bits.shape)
accuracy_eve = correct_bits_eve / total_bits * 100

print(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy by Eve: {accuracy_eve}%")

# Bob attempt to decrypt cipher1
decrypted_c1 = bob.predict([cipher1, private_arr])
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

