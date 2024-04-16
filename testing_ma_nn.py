from neural_network.networks_two_meta import create_networks
import numpy as np
from key.EllipticCurve import set_curve, generate_key_pair
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-rate', type=float, default=0.1, help='Dropout rate')
parser.add_argument('-curve', type=str, default="secp224r1", help='Elliptic curve name')
args = parser.parse_args()

curve = set_curve(args.curve)

rate = args.rate

batch_size = 448
test_type = "ma-rate-0.2-cuvre-secp224r1-batch-448-0.00005-rrandom-0-6"
print(f"Testing with {test_type}...")

# p1_batch = np.load("plaintext/p1_batch.npy")
# p2_batch = np.load("plaintext/p2_batch.npy")
# public_arr = np.load(f"key/public_key-{curve.name}.npy")
# private_arr = np.load(f"key/private_key-{curve.name}.npy")
private_arr, public_arr = generate_key_pair(batch_size, curve)

alice, bob, HO_model_addition, eve, _, _, p1_bits, _, p2_bits, _, _, nonce_bits, HO_model_multiplication = create_networks(public_arr.shape[1], private_arr.shape[1], rate)

p1_batch = np.random.randint(
    0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits).astype('float32')
p2_batch = np.random.randint(
    0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits).astype('float32')

HO_a_weights_path = f'weights/weights-{test_type}/addition_weights.h5'
HO_m_weights_path = f'weights/weights-{test_type}/multiplication_weights.h5'
alice_weights_path = f'weights/weights-{test_type}/alice_weights.h5'
bob_weights_path = f'weights/weights-{test_type}/bob_weights.h5'
eve_weights_path = f'weights/weights-{test_type}/eve_weights.h5'

HO_model_addition.load_weights(HO_a_weights_path)
HO_model_multiplication.load_weights(HO_m_weights_path)
alice.load_weights(alice_weights_path)
bob.load_weights(bob_weights_path)
eve.load_weights(eve_weights_path)

nonce = np.random.rand(batch_size, nonce_bits)

# Alice encrypts the message
cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch, nonce])
print(f"Cipher1: {cipher1}")
print(f"Cipher2: {cipher2}")
print(cipher1.shape)

# HO adds the messages
op_a = np.zeros((cipher1.shape))
cipher3_a = HO_model_addition.predict([op_a, cipher1, cipher2])
computed_cipher = cipher1 + cipher2
tolerance = 1e-3
correct_elements = np.sum(np.abs(computed_cipher - cipher3_a) <= tolerance)
total_elements = np.prod(cipher3_a.shape)
accuracy_percentage_add = (correct_elements / total_elements) * 100
print()
print("---HO model addition---")
print(f"HO model correct: {correct_elements}")
print(f"Total Elements: {total_elements}")
print(f"HO model Accuracy Percentage: {accuracy_percentage_add:.2f}%")
print(f"Cipher3: {cipher3_a}")
print(f"C1*C2: {computed_cipher}")

# HO multiplies the messages
op_m = np.ones((cipher1.shape))
cipher3_m = HO_model_multiplication.predict([op_m, cipher1, cipher2])
computed_cipher = cipher1 * cipher2
tolerance = 1e-3
correct_elements = np.sum(np.abs(computed_cipher - cipher3_m) <= tolerance)
total_elements = np.prod(cipher3_m.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print()
print("---HO model multiplication---")
print(f"HO model correct: {correct_elements}")
print(f"Total Elements: {total_elements}")
print(f"HO model Accuracy Percentage: {accuracy_percentage:.2f}%")
print(f"Cipher3: {cipher3_m}")
print(f"C1+C2: {computed_cipher}")


# Bob attempt to decrypt C3
decrypted_a = bob.predict([cipher3_a, private_arr, nonce])
decrypted_bits = np.round(decrypted_a).astype(int)

# Calculate Bob's decryption accuracy
correct_bits = np.sum(decrypted_bits == (p1_batch+p2_batch))
total_bits = np.prod(decrypted_bits.shape)
accuracy_a = correct_bits / total_bits * 100

print()
print("---Bob decryption P1+P2---")
print(f"Number of correctly decrypted bits by Bob: {correct_bits}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy Bob: {accuracy_a}%")
print(f"Bob decrypted: {decrypted_bits}")
print(f"P1+P2: {p1_batch+p2_batch}")

# Bob attempt to decrypt C3
decrypted_m = bob.predict([cipher3_m, private_arr, nonce])
decrypted_bits = np.round(decrypted_m).astype(int)

# Calculate Bob's decryption accuracy
correct_bits = np.sum(decrypted_bits == (p1_batch*p2_batch))
total_bits = np.prod(decrypted_bits.shape)
accuracy = correct_bits / total_bits * 100

# # Eve attempt to decrypt C3
eve_decrypted = eve.predict([cipher3_m, public_arr, nonce])
eve_decrypted_bits = np.round(eve_decrypted).astype(int)
print(eve_decrypted_bits)

# Calculate Eve's decryption accuracy
correct_bits_eve = np.sum(eve_decrypted_bits == (p1_batch*p2_batch))
total_bits = np.prod(eve_decrypted_bits.shape)
accuracy_eve_m = correct_bits_eve / total_bits * 100

print()
print("Eve decryption of P1+P2")
print(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy by Eve: {accuracy_eve_m}%")

print()
print("---Bob decryption P1*P2---")
print(f"Number of correctly decrypted bits by Bob: {correct_bits}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy Bob: {accuracy}%")
print(f"Bob decrypted: {decrypted_bits}")
print(f"P1*P2: {p1_batch*p2_batch}")
print()

# Eve attempt to decrypt C3
eve_decrypted = eve.predict([cipher3_a, public_arr, nonce])
eve_decrypted_bits = np.round(eve_decrypted).astype(int)


# Calculate Eve's decryption accuracy
correct_bits_eve = np.sum(eve_decrypted_bits == (p1_batch+p2_batch))
total_bits = np.prod(eve_decrypted_bits.shape)
accuracy_eve_a = correct_bits_eve / total_bits * 100

print()
print("Eve decryption of P1*P2")
print(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy by Eve: {accuracy_eve_a}%")
print(f"Eve decrypted bits: {eve_decrypted_bits}")


# # Bob attempt to decrypt C1
decrypted_c1 = bob.predict([cipher1, private_arr, nonce])
decrypted_bits_c1 = np.round(decrypted_c1).astype(int)

# Calculate Bob's decryption accuracy
correct_bits_p1 = np.sum(decrypted_bits_c1 == (p1_batch))
total_bits_p1 = np.prod(decrypted_bits_c1.shape)
accuracy_p1 = correct_bits_p1 / total_bits_p1 * 100

print()
print(f"Decryption accuracy P1: {accuracy_p1}%")
print(f" P1: {p1_batch}")
print(f"Bob decrypted bits P1: {decrypted_bits_c1}")


# Eve attempt to decrypt C1
decrypted_c1_eve = eve.predict([cipher1, public_arr, nonce])
decrypted_bits_c1_eve = np.round(decrypted_c1_eve).astype(int)

# Calculate Bob's decryption accuracy
correct_bits_p1_eve = np.sum(decrypted_bits_c1_eve == (p1_batch))
total_bits_p1_eve = np.prod(decrypted_bits_c1_eve.shape)
accuracy_p1_eve = correct_bits_p1_eve / total_bits_p1_eve * 100

print()
print("Eve decryption of P1")
print(f"Decryption accuracy P1: {accuracy_p1_eve}%")
print(f"Eve decrypted bits P1: {decrypted_bits_c1_eve}")


# Bob attempt to decrypt C2
decrypted_c2 = bob.predict([cipher2, private_arr, nonce])
decrypted_bits_c2 = np.round(decrypted_c2).astype(int)

# Calculate Bob's decryption accuracy
correct_bits_p2 = np.sum(decrypted_bits_c2 == (p2_batch))
total_bits_p2 = np.prod(decrypted_bits_c2.shape)
accuracy_p2 = correct_bits_p2 / total_bits_p2 * 100

print()
print(f"Decryption accuracy P2: {accuracy_p2}%")
print(f" P2: {p2_batch}")
print(f"Bob decrypted bits P2: {decrypted_bits_c2}")

# Eve attempt to decrypt C2
decrypted_c2_eve = eve.predict([cipher2, public_arr, nonce])
decrypted_bits_c2_eve = np.round(decrypted_c2_eve).astype(int)

# Calculate Bob's decryption accuracy
correct_bits_p2_eve = np.sum(decrypted_bits_c2_eve == (p2_batch))
total_bits_p2_eve = np.prod(decrypted_bits_c2_eve.shape)
accuracy_p2_eve = correct_bits_p2_eve / total_bits_p2_eve * 100

print()
print("Eve decryption of P2")
print(f"Decryption accuracy P2: {accuracy_p2_eve}%")
print(f"Eve decrypted bits P1: {decrypted_bits_c2_eve}")

print()
print(f"HO model Addition: {accuracy_percentage_add:.2f}%")
print(f"HO model Multiplication: {accuracy_percentage:.2f}%")

print()
print(f"Decryption accuracy Bob Addition: {accuracy_a}%")
print(f"Decryption accuracy Bob Multiplication: {accuracy}%")
print(f"Decryption accuracy P1: {accuracy_p1}%")
print(f"Decryption accuracy P2: {accuracy_p2}%")

print()
print(f"Decryption accuracy Eve Addition: {accuracy_eve_a}%")
print(f"Decryption accuracy Eve Multiplication: {accuracy_eve_m}%")
print(f"Decryption accuracy P1: {accuracy_p1_eve}%")
print(f"Decryption accuracy P2: {accuracy_p2_eve}%")
