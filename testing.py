from neural_network.networks_functions import create_networks
import numpy as np
from key.EllipticCurve import curve

dropout_rate = 0.5

p1_batch = np.load("plaintext/p1_batch.npy")
p2_batch = np.load("plaintext/p2_batch.npy")
public_arr = np.load(f"key/public_key-{curve.name}.npy")
private_arr = np.load(f"key/private_key-{curve.name}.npy")

alice, bob, HO_model, eve, _, _, _, _, _, _, _, nonce_bits = create_networks(public_arr.shape[1], private_arr.shape[1], dropout_rate)

batch_size = 512
test_type = f"rate-{dropout_rate}-curve-{curve.name}"
print(f"Testing with {test_type}...")

HO_weights_path = f'weights/weights-{test_type}/addition_weights.h5'
alice_weights_path = f'weights/weights-{test_type}/alice_weights.h5'
bob_weights_path = f'weights/weights-{test_type}/bob_weights.h5'
eve_weights_path = f'weights/weights-{test_type}/eve_weights.h5'

HO_model.load_weights(HO_weights_path)
alice.load_weights(alice_weights_path)
bob.load_weights(bob_weights_path)
eve.load_weights(eve_weights_path)

nonce = np.random.rand(batch_size, nonce_bits)

# Alice encrypts the message
cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])
print(f"Cipher1: {cipher1}")
np.save(f"ciphertext/{test_type}-1.npy", cipher1)

print(f"Cipher2: {cipher2}")

# HO adds the messages
cipher3 = HO_model.predict([cipher1, cipher2])
computed_cipher = cipher1 + cipher2
tolerance = 1e-4
correct_elements = np.sum(np.abs(computed_cipher - cipher3) <= tolerance)
total_elements = np.prod(cipher3.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print("HO model addition")
print(f"HO model correct: {correct_elements}")
print(f"Total Elements: {total_elements}")
print(f"HO model Accuracy Percentage: {accuracy_percentage:.2f}%")
# print(f"Cipher3: {cipher3}")

# Bob attempt to decrypt C3
decrypted = bob.predict([cipher3, private_arr, nonce])
decrypted_bits = np.round(decrypted).astype(int)

# Calculate Bob's decryption accuracy
correct_bits = np.sum(decrypted_bits == (p1_batch+p2_batch))
total_bits = np.prod(decrypted_bits.shape)
accuracy = correct_bits / total_bits * 100

print()
print("Bob decryption P1+P2")
print(f"Number of correctly decrypted bits by Bob: {correct_bits}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy Bob: {accuracy}%")
# print(f"Bob decrypted: {decrypted}")
# print(f"Bob decrypted bits: {decrypted_bits}")

# Eve attempt to decrypt C3
eve_decrypted = eve.predict([cipher3, public_arr, nonce])
eve_decrypted_bits = np.round(eve_decrypted).astype(int)

# Calculate Eve's decryption accuracy
correct_bits_eve = np.sum(eve_decrypted_bits == (p1_batch+p2_batch))
total_bits = np.prod(eve_decrypted_bits.shape)
accuracy_eve = correct_bits_eve / total_bits * 100

print()
print("Eve decryption of P1+P2")
print(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy by Eve: {accuracy_eve}%")
# print(f"Eve decrypted: {eve_decrypted}")
# print(f"Eve decrypted bits: {eve_decrypted_bits}")


# Bob attempt to decrypt C1
decrypted_c1 = bob.predict([cipher1, private_arr, nonce])
decrypted_bits_c1 = np.round(decrypted_c1).astype(int)

# Calculate Bob's decryption accuracy
correct_bits_p1 = np.sum(decrypted_bits_c1 == (p1_batch))
total_bits_p1 = np.prod(decrypted_bits_c1.shape)
accuracy_p1 = correct_bits_p1 / total_bits_p1 * 100

print()
print("Bob decryption of P1")
print(f"Number of correctly decrypted bits P1: {correct_bits_p1}")
print(f"Total number of bits P1: {total_bits_p1}")
print(f"Decryption accuracy P1: {accuracy_p1}%")
# print(f"Bob decrypted P1: {decrypted_c1}")
# print(f"Bob decrypted bits P1: {decrypted_bits_c1}")


# Eve attempt to decrypt C1
decrypted_c1_eve = eve.predict([cipher1, public_arr, nonce])
decrypted_bits_c1_eve = np.round(decrypted_c1_eve).astype(int)

# Calculate Bob's decryption accuracy
correct_bits_p1_eve = np.sum(decrypted_bits_c1_eve == (p1_batch))
total_bits_p1_eve = np.prod(decrypted_bits_c1_eve.shape)
accuracy_p1_eve = correct_bits_p1_eve / total_bits_p1_eve * 100

print()
print("Eve decryption of P1")
print(f"Number of correctly decrypted bits P1: {correct_bits_p1_eve}")
print(f"Total number of bits P1: {total_bits_p1_eve}")
print(f"Decryption accuracy P1: {accuracy_p1_eve}%")
# print(f"Eve decrypted P1: {decrypted_c1_eve}")
# print(f"Eve decrypted bits P1: {decrypted_bits_c1_eve}")


# Bob attempt to decrypt C2
decrypted_c2 = bob.predict([cipher2, private_arr, nonce])
decrypted_bits_c2 = np.round(decrypted_c2).astype(int)

# Calculate Bob's decryption accuracy
correct_bits_p2 = np.sum(decrypted_bits_c2 == (p2_batch))
total_bits_p2 = np.prod(decrypted_bits_c2.shape)
accuracy_p2 = correct_bits_p2 / total_bits_p2 * 100

print()
print("Bob decryption P2")
print(f"Number of correctly decrypted bits P2: {correct_bits_p2}")
print(f"Total number of bits P2: {total_bits_p2}")
print(f"Decryption accuracy P2: {accuracy_p2}%")
# print(f"Bob decrypted P2: {decrypted_c2}")
# print(f"Bob decrypted bits P2: {decrypted_bits_c2}")

# Eve attempt to decrypt C2
decrypted_c2_eve = eve.predict([cipher2, public_arr, nonce])
decrypted_bits_c2_eve = np.round(decrypted_c2_eve).astype(int)

# Calculate Bob's decryption accuracy
correct_bits_p2_eve = np.sum(decrypted_bits_c2_eve == (p2_batch))
total_bits_p2_eve = np.prod(decrypted_bits_c2_eve.shape)
accuracy_p2_eve = correct_bits_p2_eve / total_bits_p2_eve * 100

print()
print("Eve decryption of P2")
print(f"Number of correctly decrypted bits P2: {correct_bits_p2_eve}")
print(f"Total number of bits P2: {total_bits_p2_eve}")
print(f"Decryption accuracy P2: {accuracy_p2_eve}%")
# print(f"Eve decrypted P1: {decrypted_c2_eve}")
# print(f"Eve decrypted bits P1: {decrypted_bits_c2_eve}")
