from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Lambda, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from nalu import NALU
from nac import NAC
from data_utils_combined import generate_static_dataset
import numpy as np

task_m = lambda x, y: x * y
task_a = lambda x, y: x + y

c1_bits = 16
c2_bits = 16
c3_bits = 16
num_samples = c3_bits
batch_size = 512
nonce_bits = 64

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

optimizer = Adam(0.02)
HO_model.compile(optimizer, 'mse')

HO_model.trainable = True

# Train HO model to do addition
X1_train, X2_train, y_train = generate_static_dataset(task_m, task_a, num_samples, batch_size)
X1_test, X2_test, y_test = generate_static_dataset(task_m, task_a, num_samples, batch_size)

HO_model.fit([X1_train, X2_train], y_train, batch_size=512, epochs=5000,
    verbose=2, validation_data=([X1_test, X2_test], y_test))

HO_model.trainable = False

x1, x2, y = generate_static_dataset(lambda x, y: x * y, lambda x, y: x + y, 16, 512)
print(x1)
print(x2)
print(y)
cipher3 = HO_model.predict([x1, x2])
tolerance = 1e-4
correct_elements = np.sum(np.abs(y - cipher3) <= tolerance)
total_elements = np.prod(cipher3.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print("HO model addition")
print(f"HO model correct: {correct_elements}")
print(f"Total Elements: {total_elements}")
print(f"HO model Accuracy Percentage: {accuracy_percentage:.2f}%")
print(f"Cipher3: {cipher3}")