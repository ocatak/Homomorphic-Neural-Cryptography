from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Lambda, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from key.EllipticCurve import get_key_shape, set_curve
from neural_network.nalu import NALU
from tensorflow.keras.models import Model
from typing import Tuple


def process_plaintext(ainput0, ainput1, anonce_input, p_bits, public_bits, nonce_bits, dropout_rate, pad):
    """Alice network to process plaintexts.

    Args:
        ainput0: Public key input.
        ainput1: Plaintext input.
        anonce_input: Nonce input.
        p_bits: Number of bits in the plaintext.
        public_bits: Number of bits in the public key.
        nonce_bits: Number of bits in the nonce.
        dropout_rate: Dropout rate.
        pad: Padding type.
    
    Returns:
        The output of the Alice network.
    """
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
                    padding=pad, activation='sigmoid')(aconv3)

    return Flatten()(aconv4)


def create_networks(public_bits: int, private_bits: int, dropout_rate: float
) -> Tuple[Model, Model, Model, Model, Model, int, int, Model, int, float, int, int]:
    """Creates the Alice, Bob, HO and Eve networks.
    
    Args:
        public_bits: Number of bits in the public key.
        private_bits: Number of bits in the private key.
        dropout_rate: Dropout rate.
    
    Returns: 
        alice, bob, HO_model, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, learning_rate, c3_bits and nonce_bits, used for training
    """
    learning_rate = 0.0001
    
    nonce_bits = 64

    p1_bits = 16
    p2_bits = 16

    c1_bits = (p1_bits+public_bits+nonce_bits)//2 
    c2_bits = (p2_bits+public_bits+nonce_bits)//2 
    c3_bits = (c1_bits+c2_bits)//2

    pad = 'same'

    # Size of the message space
    m_train = 2**((p1_bits+p2_bits)//2)

    # Define Alice inputs
    ainput0 = Input(shape=(public_bits,))       # public key
    ainput1 = Input(shape=(p1_bits))            # plaintext 1
    ainput2 = Input(shape=(p2_bits))            # plaintext 2
    anonce_input = Input(shape=(nonce_bits))    # nonce

    aoutput_first = process_plaintext(ainput0, ainput1, anonce_input, p1_bits, public_bits, nonce_bits, dropout_rate, pad)
    aoutput_second = process_plaintext(ainput0, ainput2, anonce_input, p2_bits, public_bits, nonce_bits, dropout_rate, pad)

    alice = Model(inputs=[ainput0, ainput1, ainput2, anonce_input],
                outputs=[aoutput_first, aoutput_second], name='alice')


    # Generate the HO_model network
    units = 2
    HOinput0 = Input(shape=(c3_bits))  # multiplication or addition
    HOinput1 = Input(shape=(c1_bits))  # ciphertext 1
    HOinput2 = Input(shape=(c2_bits))  # ciphertext 2

    HO_reshape0 = Reshape((c3_bits, 1))(HOinput0)
    HO_reshape1 = Reshape((c1_bits, 1))(HOinput1)
    HO_reshape2 = Reshape((c2_bits, 1))(HOinput2)

    HOinput =  concatenate([HO_reshape0, HO_reshape1, HO_reshape2], axis=-1)
    x = NALU(units, use_gating=True)(HOinput)
    x = NALU(1, use_gating=True)(x)
    x = Reshape((c3_bits,))(x)

    HO_model = Model(inputs=[HOinput0, HOinput1, HOinput2], outputs=x)


    # Bob network
    binput0 = Input(shape=(c3_bits,))           # ciphertext
    binput1 = Input(shape=(private_bits,))      # private key
    bnonce_input = Input(shape=(nonce_bits))    # nonce

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
                    padding=pad, activation='sigmoid')(bconv3)

    bflattened = Flatten()(bconv4)

    # Scale the output from [0, 1] to [0, 2] by multiplying by 2
    boutput = Lambda(lambda x: x * 2)(bflattened)

    bob = Model(inputs=[binput0, binput1, bnonce_input],
                outputs=boutput, name='bob')


    # Eve network
    einput0 = Input(shape=(c3_bits,))           # ciphertext
    einput1 = Input(shape=(public_bits, ))      # public key
    enonce_input = Input(shape=(nonce_bits))    # nonce

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
                    padding=pad, activation='sigmoid')(econv3)

    eflattened = Flatten()(econv4)

    eoutput = Lambda(lambda x: x * 2)(eflattened)

    eve = Model([einput0, einput1, enonce_input], eoutput, name='eve')


    # Loss and optimizer

    aliceout1, aliceout2 = alice([ainput0, ainput1, ainput2, anonce_input]) # Alice gets two outputs from 3 inputs

    HOout = HO_model([HOinput0, aliceout1, aliceout2])  # HO_model get one output from Alice's two output

    # Eve and bob get one output from HO_model output with the size of p1+p2
    bobout = bob([HOout, binput1, anonce_input]) 
    eveout = eve([HOout, ainput0, anonce_input])

    # Eve and Bob output from alice to decrypt p1/p2
    bobout_alice = bob([aliceout1, binput1, anonce_input])
    eveout_alice = eve([aliceout1, ainput0, anonce_input])

    bobout_alice2 = bob([aliceout2, binput1, anonce_input])
    eveout_alice2 = eve([aliceout2, ainput0, anonce_input])


    abhemodel = Model([ainput0, ainput1, ainput2, anonce_input, binput1, HOinput0],
                    bobout, name='abhemodel')

    # Loss functions
    eveloss_addition = K.mean(K.sum(K.abs(ainput1 + ainput2 - eveout), axis=-1))
    bobloss_addition = K.mean(K.sum(K.abs(ainput1 + ainput2 - bobout), axis=-1))

    eveloss_multiplication = K.mean(K.sum(K.abs(ainput1 * ainput2 - eveout), axis=-1))
    bobloss_multiplication = K.mean(K.sum(K.abs(ainput1 * ainput2 - bobout), axis=-1))

    eveloss_alice = K.mean(K.sum(K.abs(ainput1 - eveout_alice), axis=-1))
    bobloss_alice = K.mean(K.sum(K.abs(ainput1 - bobout_alice), axis=-1))

    eveloss_alice2 = K.mean(K.sum(K.abs(ainput2 - eveout_alice2), axis=-1))
    bobloss_alice2 = K.mean(K.sum(K.abs(ainput2 - bobout_alice2), axis=-1))

    eveloss = (eveloss_addition+eveloss_multiplication+eveloss_alice+eveloss_alice2)/4
    bobloss = (bobloss_addition+bobloss_multiplication+bobloss_alice+bobloss_alice2)/4

    # eveloss = (eveloss_addition + eveloss_multiplication)/2
    # bobloss = (bobloss_addition + bobloss_multiplication)/2

    # Initial weights based on assumption that multiplication is harder
    # weight_addition = 1
    # weight_multiplication = 1

    # eveloss = (weight_addition * eveloss_addition + weight_multiplication * eveloss_multiplication) / (weight_addition + weight_multiplication)
    # bobloss = (weight_addition * bobloss_addition + weight_multiplication * bobloss_multiplication) / (weight_addition + weight_multiplication)

    # eveloss_a = (eveloss_alice + eveloss_alice2)/2
    # bobloss_a = (bobloss_alice + bobloss_alice2)/2

    # eveloss = (eveloss_a + eveloss)/2
    # bobloss = (bobloss_a + bobloss)/2

    # Build and compile the ABHE model, used for training Alice, Bob and HE networks
    abheloss = bobloss + K.square((p1_bits+p2_bits)/2 - eveloss) / ((p1_bits+p2_bits//2)**2)
    abhemodel.add_loss(abheloss)

    beoptim = Adam(learning_rate=learning_rate)
    eveoptim = Adam(learning_rate=learning_rate)
    optimizer = Adam(0.1)
    HO_model.compile(optimizer, 'mse')
    abhemodel.compile(optimizer=beoptim)


    # Build and compile the Eve model, used for training Eve net (with Alice frozen)
    alice.trainable = False
    evemodel = Model([ainput0, ainput1, ainput2, anonce_input, HOinput0], eveout, name='evemodel')
    evemodel.add_loss(eveloss)
    evemodel.compile(optimizer=eveoptim)

    return alice, bob, HO_model, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, learning_rate, c3_bits, nonce_bits

if __name__ == "__main__":
    curve = set_curve("secp256r1")
    public_bits = get_key_shape(curve)[1]  
    private_bits = get_key_shape(curve)[0]
    dropout_rate = 0.6
    create_networks(public_bits, private_bits, dropout_rate) 