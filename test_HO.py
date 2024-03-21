from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Lambda, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from key.EllipticCurve import get_key_shape, set_curve
from neural_network.nalu import NALU
from data_utils.dataset_generator import generate_static_dataset, generate_cipher_dataset

c1_bits = 2
c2_bits = 2
c3_bits = 2
task_fn = lambda x, y: x * y
batch_size = 512

units = 2
HOinput1 = Input(shape=(c1_bits))  # ciphertext 1
HOinput2 = Input(shape=(c2_bits))  # ciphertext 2

HO_reshape1 = Reshape((c1_bits, 1))(HOinput1)
HO_reshape2 = Reshape((c2_bits, 1))(HOinput2)

HOinput =  concatenate([HO_reshape1, HO_reshape2], axis=-1)
x = NALU(units)(HOinput)
x = NALU(1)(x)
x = Reshape((c3_bits,))(x)

HO_model = Model(inputs=[HOinput1, HOinput2], outputs=x)

optimizer = RMSprop(0.1)
HO_model.compile(optimizer, 'mse')

HO_model.trainable = True

X1_train, X2_train, y_train = generate_static_dataset(task_fn, c3_bits, batch_size)
X1_test, X2_test, y_test = generate_static_dataset(task_fn, c3_bits, batch_size)

HO_model.fit([X1_train, X2_train], y_train, batch_size=128, epochs=512,
    verbose=2, validation_data=([X1_test, X2_test], y_test))

HO_model.trainable = False

predicted = HO_model.predict([X1_test, X2_test], 128)
print(X1_test[:1])
print(X2_test[:1])
print(y_test[:1])
print(predicted[:1])