from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Lambda, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, Adam
from key.EllipticCurve import get_key_shape, set_curve
from neural_network.nalu import NALU
from neural_network.nac import NAC
from tensorflow.keras.activations import relu

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
    learning_rate = 0.0001  # Adam and 0.0008

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



    # Generate the HO_model network with an input layer and two NAC layers for addition
    units = 2
    HOinput0_addition = Input(shape=(c3_bits)) # operation
    HOinput1_addition = Input(shape=(c1_bits))  # ciphertext 1
    HOinput2_addition = Input(shape=(c2_bits))  # ciphertext 2

    HO_reshape0_addition = Reshape((c3_bits, 1))(HOinput0_addition)
    HO_reshape1_addition = Reshape((c1_bits, 1))(HOinput1_addition)
    HO_reshape2_addition = Reshape((c2_bits, 1))(HOinput2_addition)

    HOinput_addition =  concatenate([HO_reshape0_addition, HO_reshape1_addition, HO_reshape2_addition], axis=-1)
    x_a = NAC(units)(HOinput_addition)
    x_a = NAC(1)(x_a)
    x_a = Reshape((c3_bits,))(x_a)

    HO_model_addition = Model(inputs=[HOinput0_addition, HOinput1_addition, HOinput2_addition], outputs=x_a)

    # Generate the HO_model network with an input layer and two NALU layers for multiplication
    units = 2
    HOinput0_multiplication = Input(shape=(c3_bits)) # operation
    HOinput1_multiplication = Input(shape=(c1_bits))  # ciphertext 1
    HOinput2_multiplication = Input(shape=(c2_bits))  # ciphertext 2

    HO_reshape0_multiplication = Reshape((c3_bits, 1))(HOinput0_multiplication)
    HO_reshape1_multiplication = Reshape((c1_bits, 1))(HOinput1_multiplication)
    HO_reshape2_multiplication = Reshape((c2_bits, 1))(HOinput2_multiplication)

    HOinput_multiplication =  concatenate([HO_reshape0_multiplication, HO_reshape1_multiplication, HO_reshape2_multiplication], axis=-1)
    x_m = NALU(units)(HOinput_multiplication)
    x_m = NALU(1)(x_m)
    x_m = Reshape((c3_bits,))(x_m)

    HO_model_multiplication = Model(inputs=[HOinput0_multiplication, HOinput1_multiplication, HOinput2_multiplication], outputs=x_m)

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
    HOout_addition = HO_model_addition([HOinput0_addition, aliceout1, aliceout2])
    HOout_multiplication = HO_model_addition([HOinput0_multiplication, aliceout1, aliceout2])

    # Eve and bob get one output from HO_model output with the size of p1+p2
    bobout_addition = bob([HOout_addition, binput1, anonce_input]) 
    eveout_addition = eve([HOout_addition, ainput0, anonce_input])

    # Eve and bob get one output from HO_model output with the size of p1*p2
    bobout_multiplication = bob([HOout_multiplication, binput1, anonce_input]) 
    eveout_multiplication = eve([HOout_multiplication, ainput0, anonce_input])

    # Eve and Bob output from alice to decrypt p1/p2
    bobout_alice1 = bob([aliceout1, binput1, anonce_input])
    eveout_alice1 = eve([aliceout1, ainput0, anonce_input])

    bobout_alice2 = bob([aliceout2, binput1, anonce_input])
    eveout_alice2 = eve([aliceout2, ainput0, anonce_input])

    abhemodel = Model([ainput0, ainput1, ainput2, anonce_input, binput1, HOinput0_addition, HOinput0_multiplication],
                    [bobout_addition, bobout_multiplication], name='abhemodel')

    # Loss functions
    eveloss_addition = K.mean(K.sum(K.abs(ainput1 + ainput2 - eveout_addition), axis=-1))
    bobloss_addition = K.mean(K.sum(K.abs(ainput1 + ainput2 - bobout_addition), axis=-1))

    eveloss_multiplication = K.mean(K.sum(K.abs(ainput1 * ainput2 - eveout_multiplication), axis=-1))
    bobloss_multiplication = K.mean(K.sum(K.abs(ainput1 * ainput2 - bobout_multiplication), axis=-1))

    eveloss_alice = K.mean(K.sum(K.abs(ainput1 - eveout_alice1), axis=-1))
    bobloss_alice = K.mean(K.sum(K.abs(ainput1 - bobout_alice1), axis=-1))

    eveloss_alice2 = K.mean(K.sum(K.abs(ainput2 - eveout_alice2), axis=-1))
    bobloss_alice2 = K.mean(K.sum(K.abs(ainput2 - bobout_alice2), axis=-1))

    eveloss = (eveloss_addition+eveloss_multiplication+eveloss_alice+eveloss_alice2)/4
    bobloss = (bobloss_addition+bobloss_multiplication+bobloss_alice+bobloss_alice2)/4

    # Build and compile the ABHE model, used for training Alice, Bob and HE networks
    abheloss = bobloss + K.square((p1_bits+p2_bits)/2 - eveloss) / ((p1_bits+p2_bits//2)**2)
    abhemodel.add_loss(abheloss)

    # Set the Adam optimizer
    beoptim = Adam(learning_rate=learning_rate)
    eveoptim = Adam(learning_rate=learning_rate)
    optimizer = RMSprop(0.1)
    HO_model_addition.compile(optimizer, 'mse')
    HO_model_multiplication.compile(optimizer, 'mse')
    abhemodel.compile(optimizer=beoptim)

    # Build and compile the Eve model, used for training Eve net (with Alice frozen)
    alice.trainable = False
    evemodel = Model([ainput0, ainput1, ainput2, anonce_input, HOinput0_addition, HOinput0_multiplication], [eveout_addition, eveout_multiplication], name='evemodel')
    evemodel.add_loss(eveloss)
    evemodel.compile(optimizer=eveoptim)

    return alice, bob, HO_model_addition, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, learning_rate, c3_bits, nonce_bits, HO_model_multiplication

if __name__ == "__main__":
    # Public and private key, changed to fit the key generated in EllipticCurve.py
    curve = set_curve("secp256r1")
    public_bits = get_key_shape(curve)[1]  
    private_bits = get_key_shape(curve)[0]
    dropout_rate = 0.6
    create_networks(public_bits, private_bits, dropout_rate) 