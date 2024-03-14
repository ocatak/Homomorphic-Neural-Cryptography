from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Lambda, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from key.EllipticCurve import get_key_shape
from nalu import NALU
from nac import NAC

# Process plaintexts
def process_plaintext(ainput0, ainput1, anonce_input, p_bits, public_bits, nonce_bits, dropout_rate, pad):
    ainput = concatenate([ainput0, ainput1, anonce_input], axis=1)

    adense1 = Dense(units=(p_bits + public_bits + nonce_bits), activation='tanh')(ainput)

    dropout = Dropout(dropout_rate)(adense1, training=True)

    areshape = Reshape((p_bits + public_bits + nonce_bits, 1,))(dropout)

    aconv1 = Conv1D(filters=2, kernel_size=4, strides=1,
                    padding=pad, activation='tanh')(areshape)

    aconv2 = Conv1D(filters=4, kernel_size=2, strides=2,
                    padding=pad, activation='tanh')(aconv1)

    aconv3 = Conv1D(filters=4, kernel_size=1, strides=1,
                    padding=pad, activation='tanh')(aconv2)

    aconv4 = Conv1D(filters=1, kernel_size=1, strides=1,
                    padding=pad, activation='hard_sigmoid')(aconv3)

    return Flatten()(aconv4)

# Alice network
def create_networks(public_bits, private_bits, dropout_rate):
    learning_rate = 0.0001

    # Set up the crypto parameters: plaintext, key, and ciphertext bit lengths
    # Plaintext 1 and 2
    p1_bits = 16
    p2_bits = 16

    # nonce bits
    nonce_bits = 64

    # Ciphertext 1 and 2
    c1_bits = (p1_bits+public_bits+nonce_bits)//2 
    c2_bits = (p2_bits+public_bits+nonce_bits)//2 

    c3_bits = (c1_bits+c2_bits)//2

    pad = 'same'

    # Size of the message space
    m_train = 2**((p1_bits+p2_bits)//2) # mabye add p2_bits

    # Define Alice inputs
    ainput0 = Input(shape=(public_bits,))  # public key
    ainput1 = Input(shape=(p1_bits))  # plaintext 1
    ainput2 = Input(shape=(p2_bits))  # plaintext 2
    anonce_input = Input(shape=(nonce_bits))  # nonce

    aoutput_first = process_plaintext(ainput0, ainput1, anonce_input, p1_bits, public_bits, nonce_bits, dropout_rate, pad)
    aoutput_second = process_plaintext(ainput0, ainput2, anonce_input, p2_bits, public_bits, nonce_bits, dropout_rate, pad)

    alice = Model(inputs=[ainput0, ainput1, ainput2, anonce_input],
                outputs=[aoutput_first, aoutput_second], name='alice')



    # Generate the HO_model network with an input layer and two NAC layers
    units = 2
    HOinput1 = Input(shape=(c1_bits))  # ciphertext 1
    HOinput2 = Input(shape=(c2_bits))  # ciphertext 2

    HO_reshape1 = Reshape((c1_bits, 1))(HOinput1)
    HO_reshape2 = Reshape((c2_bits, 1))(HOinput2)

    HOinput =  concatenate([HO_reshape1, HO_reshape2], axis=-1)
    nalu1 = NALU(units)(HOinput)
    nalu2 = NALU(1)(nalu1)
    nalu_combined = concatenate([nalu2, HO_reshape1])
    nac1 = NAC(units)(nalu_combined)
    nac2 = NAC(1)(nac1)
    nac_reshaped = Reshape((c3_bits,))(nac2)

    HO_model = Model(inputs=[HOinput1, HOinput2], outputs=nac_reshaped)

    # Bob network
    binput0 = Input(shape=(c3_bits,))  # Input will be of shape c3
    binput1 = Input(shape=(private_bits,))  # private key
    bnonce_input = Input(shape=(nonce_bits))  # nonce

    binput = concatenate([binput0, binput1, bnonce_input], axis=1)

    bdense1 = Dense(units=((p1_bits+p2_bits)), activation='tanh')(binput)
    breshape = Reshape(((p1_bits+p2_bits), 1,))(bdense1)

    bconv1 = Conv1D(filters=2, kernel_size=4, strides=1,
                    padding=pad, activation='tanh')(breshape)
    bconv2 = Conv1D(filters=4, kernel_size=2, strides=2,
                    padding=pad, activation='tanh')(bconv1)
    bconv3 = Conv1D(filters=4, kernel_size=1, strides=1,
                    padding=pad, activation='tanh')(bconv2)
    bconv4 = Conv1D(filters=1, kernel_size=1, strides=1,
                    padding=pad, activation='hard_sigmoid')(bconv3)

    # Output corresponding to shape of p1 + p2
    bflattened = Flatten()(bconv4)

    # Scale the output from [0, 1] to [0, 2] by multiplying by 2
    boutput = Lambda(lambda x: x * 2)(bflattened)

    bob = Model(inputs=[binput0, binput1, bnonce_input],
                outputs=boutput, name='bob')


    # Eve network
    einput0 = Input(shape=(c3_bits,))  # Input will be of shape c3
    einput1 = Input(shape=(public_bits, )) # public key
    enonce_input = Input(shape=(nonce_bits))  # nonce


    einput = concatenate([einput0, einput1, enonce_input], axis=1)

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
                    padding=pad, activation='hard_sigmoid')(econv3)

    # Eve's attempt at guessing the plaintext, corresponding to shape of p1 + p2
    eflattened = Flatten()(econv4)

    eoutput = Lambda(lambda x: x * 2)(eflattened)

    eve = Model([einput0, einput1, enonce_input], eoutput, name='eve')

    # Loss and optimizer

    # Alice gets two outputs from 3 inputs
    aliceout1, aliceout2 = alice([ainput0, ainput1, ainput2, anonce_input])

    # HO_model get one output from Alice's two output
    HOout = HO_model([aliceout1, aliceout2])

    # Eve and bob get one output from HO_model output with the size of p1+p2
    bobout = bob([HOout, binput1, anonce_input]) 
    eveout = eve([HOout, ainput0, anonce_input])

    # Eve and Bob output from alice to decrypt p1/p2
    bobout_alice1 = bob([aliceout1, binput1, anonce_input])
    eveout_alice1 = eve([aliceout1, ainput0, anonce_input])

    bobout_alice2 = bob([aliceout1, binput1, anonce_input])
    eveout_alice2 = eve([aliceout1, ainput0, anonce_input])

    abhemodel = Model([ainput0, ainput1, ainput2, anonce_input, binput1],
                    bobout, name='abhemodel')

    # Loss functions
    eveloss_ho = K.mean(K.sum(K.abs(ainput1 * ainput2 + ainput1 - eveout), axis=-1))
    bobloss_ho = K.mean(K.sum(K.abs(ainput1 * ainput2 + ainput1 - bobout), axis=-1))

    eveloss_alice1 = K.mean(K.sum(K.abs(ainput1 - eveout_alice1), axis=-1))
    bobloss_alice1 = K.mean(K.sum(K.abs(ainput1 - bobout_alice1), axis=-1))

    eveloss_alice2 = K.mean(K.sum(K.abs(ainput2 - eveout_alice2), axis=-1))
    bobloss_alice2 = K.mean(K.sum(K.abs(ainput2 - bobout_alice2), axis=-1))

    eveloss_alice = (eveloss_alice1+eveloss_alice2)/2
    bobloss_alice = (bobloss_alice1+bobloss_alice2)/2

    eveloss = (eveloss_ho + eveloss_alice)/2
    bobloss = (bobloss_ho + bobloss_alice)/2

    eveloss = (eveloss_ho + eveloss_alice)/2
    bobloss = (bobloss_ho + bobloss_alice)/2

    # Build and compile the ABHE model, used for training Alice, Bob and HE networks
    # abheloss = bobloss + K.square((p1_bits+p2_bits)/2 - eveloss) / ((p1_bits+p2_bits//2)**2)
    K = 8
    abheloss = bobloss + (K-eveloss) * ((256 - 32 * eveloss + eveloss ** 2) / 256)
    abhemodel.add_loss(abheloss)

    # Set the Adam optimizer
    beoptim = Adam(learning_rate=learning_rate)
    eveoptim = Adam(learning_rate=learning_rate)
    optimizer = RMSprop(0.1)
    HO_model.compile(optimizer, 'mse')
    abhemodel.compile(optimizer=beoptim)

    # Build and compile the Eve model, used for training Eve net (with Alice frozen)
    alice.trainable = False
    evemodel = Model([ainput0, ainput1, ainput2, anonce_input], eveout, name='evemodel')
    evemodel.add_loss(eveloss)
    evemodel.compile(optimizer=eveoptim)

    return alice, bob, HO_model, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, learning_rate, c3_bits, nonce_bits

if __name__ == "__main__":
    # Public and private key, changed to fit the key generated in EllipticCurve.py
    public_bits = get_key_shape()[1]  
    private_bits = get_key_shape()[0]
    dropout_rate = 0.6
    create_networks(public_bits, private_bits, dropout_rate) 