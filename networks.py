from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Embedding
from tensorflow.keras.optimizers import Adam
from EllipticCurve import get_key_shape
from nac import NAC


# Set up the crypto parameters: plaintext, key, and ciphertext bit lengths
# Plaintext 1 and 2
p1_bits = 8  
p2_bits = 8

# Public and private key, changed to fit the key generated in EllipticCurve.py
public_bits = get_key_shape()[1]  
private_bits = get_key_shape()[0] 

# Ciphertext 1 and 2
c1_bits = (p1_bits+public_bits)//2 
c2_bits = (p2_bits+public_bits)//2 

c3_bits = (c1_bits+c2_bits)//2

pad = 'same'

# Size of the message space
m_train = 2**(p1_bits) # mabye add p2_bits

# Alice network
# Define Alice inputs
ainput0 = Input(shape=(public_bits,))  # public key
ainput1 = Input(shape=(p1_bits))  # plaintext 1
ainput2 = Input(shape=(p2_bits))  # plaintext 2

# Process plaintexts
def process_plaintext(ainput0, ainput1, p_bits, public_bits):
    ainput = concatenate([ainput0, ainput1], axis=1)

    adense1 = Dense(units=(p_bits + public_bits), activation='tanh')(ainput)
    areshape = Reshape((p_bits + public_bits, 1,))(adense1)

    aconv1 = Conv1D(filters=2, kernel_size=4, strides=1,
                    padding=pad, activation='tanh')(areshape)

    aconv2 = Conv1D(filters=4, kernel_size=2, strides=2,
                    padding=pad, activation='tanh')(aconv1)

    aconv3 = Conv1D(filters=4, kernel_size=1, strides=1,
                    padding=pad, activation='tanh')(aconv2)

    aconv4 = Conv1D(filters=1, kernel_size=1, strides=1,
                    padding=pad, activation='sigmoid')(aconv3)

    return Flatten()(aconv4)

aoutput_first = process_plaintext(ainput0, ainput1, p1_bits, public_bits)
aoutput_second = process_plaintext(ainput0, ainput2, p2_bits, public_bits)

alice = Model(inputs=[ainput0, ainput1, ainput2],
              outputs=[aoutput_first, aoutput_second], name='alice')

# Generate the HO network with an input layer and two NAC layers
units = 2
# ip = Input(shape=(c3_bits, 2,)) # Define 2 inputs of size c1_bits
HOinput1 = Input(shape=(c1_bits))  # ciphertext 1
HOinput2 = Input(shape=(c2_bits))  # ciphertext 2

HO_reshape1 = Reshape((c1_bits, 1))(HOinput1)
HO_reshape2 = Reshape((c2_bits, 1))(HOinput2)

HOinput =  concatenate([HO_reshape1, HO_reshape2], axis=-1)
x = NAC(units)(HOinput)
x = NAC(1)(x)
x = Reshape((c3_bits,))(x)

HO = Model(inputs=[HOinput1, HOinput2], outputs=x)

# Bob network
binput0 = Input(shape=(c3_bits,))  # Input will be of shape c3
binput1 = Input(shape=(public_bits,))  # private key

binput = concatenate([binput0, binput1], axis=1)

bdense1 = Dense(units=((p1_bits+p2_bits)), activation='tanh')(binput)
breshape = Reshape(((p1_bits+p2_bits), 1,))(bdense1)

bconv1 = Conv1D(filters=2, kernel_size=4, strides=1,
                padding=pad, activation='tanh')(breshape)
bconv2 = Conv1D(filters=4, kernel_size=2, strides=2,
                padding=pad, activation='tanh')(bconv1)
bconv3 = Conv1D(filters=4, kernel_size=1, strides=1,
                padding=pad, activation='tanh')(bconv2)
bconv4 = Conv1D(filters=1, kernel_size=1, strides=1,
                padding=pad, activation='sigmoid')(bconv3)

# Output corresponding to shape of p1 + p2
boutput = Flatten()(bconv4)


bob = Model(inputs=[binput0, binput1],
            outputs=boutput, name='bob')

# Eve network
einput = Input(shape=(c3_bits,))  # Input will be of shape c3

edense1 = Dense(units=((p1_bits+p2_bits)), activation='tanh')(einput)
edense2 = Dense(units=((p1_bits+p2_bits)), activation='tanh')(edense1)
ereshape = Reshape(((p1_bits+p2_bits), 1,))(edense2)

econv1 = Conv1D(filters=2, kernel_size=4, strides=1,
                padding=pad, activation='tanh')(ereshape)
econv2 = Conv1D(filters=4, kernel_size=2, strides=2,
                padding=pad, activation='tanh')(econv1)
econv3 = Conv1D(filters=4, kernel_size=1, strides=1,
                padding=pad, activation='tanh')(econv2)
econv4 = Conv1D(filters=1, kernel_size=1, strides=1,
                padding=pad, activation='sigmoid')(econv3)

# Eve's attempt at guessing the plaintext, corresponding to shape of p1 + p2
eoutput = Flatten()(econv4)

eve = Model(einput, eoutput, name='eve')


# Loss and optimizer

# Alice gets two outputs from 3 inputs
aliceout1, aliceout2 = alice([ainput0, ainput1, ainput2])

# HO get one output from Alice's two output
HOout = HO([aliceout1, aliceout2])

# Eve and bob get one output from HO output with the size of p1+p2
bobout = bob([HOout, binput1]) 
eveout = eve(HOout)


# #! Loss functions need to change
eveloss = K.mean(K.sum(K.abs(ainput1 + ainput2 - eveout), axis=-1))
bobloss = K.mean(K.sum(K.abs(ainput1 + ainput2 - bobout), axis=-1))

