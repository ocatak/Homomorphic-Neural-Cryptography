from networks import alice, p1_bits, p2_bits
import numpy as np
from EllipticCurve import generate_key_pair
from keras.models import Model
from keras.layers import Input, Reshape, concatenate
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import numpy as np

import os
import sys
module_dir = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, module_dir)
from nac import NAC
from data_utils import generate_static_dataset

# Check if weights folder exists
# This folder store the trained HO_model weights
if not os.path.exists('weights'):
    os.makedirs('weights')


c1_bits = 572
c2_bits = 572 

c3_bits = 572

# Hyper parameters used for training
units = 2
batch_size = 5
num_samples = 572
# num_samples = 864*batch_size #rows

# Create a task function for addition
task_name = 'addition'
task_fn = lambda x, y: x + y

# Generate the HO_model network with an input layer and two NAC layers
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

HO_model = Model(inputs=[HOinput1, HOinput2], outputs=x)

# Compile the HO_model
# Use RMSprop as the optimizer and mean squared error as the loss function
optimizer = RMSprop(0.1)
HO_model.compile(optimizer, 'mse')

# Generate training and testing datasets
X1_train, X2_train, y_train = generate_static_dataset(task_fn, num_samples, batch_size, mode='interpolation')

private_arr, public_arr = generate_key_pair(batch_size)

p1_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
p2_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch])

y_test = []
assert callable(task_fn)
for i in range(len(cipher1)):
    Y = task_fn(cipher1[i], cipher2[i])
    y_test.append(Y)

y_test = np.array(y_test)


# Use 'ModelCheckpoint' callback to save the HO_model weights during training, specifically saving only the best-performing weights based on validation loss
weights_path = 'weights/%s_weights.h5' % (task_name)
checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                             verbose=1, save_weights_only=True, save_best_only=True)

callbacks = [checkpoint]

# Train HO_model and includes validation using the test dataset
HO_model.fit([X1_train, X2_train], y_train, batch_size=64, epochs=500,
          verbose=2, callbacks=callbacks, validation_data=([cipher1, cipher2], y_test))

# Evaluate the HO_model on the test data set and the mean squared error of the HO_model on the test dataset is printed
HO_model.load_weights(weights_path)

scores = HO_model.evaluate([cipher1, cipher2], y_test, batch_size=128)
predicted = HO_model.predict([cipher1, cipher2], 128)
print([cipher1, cipher2][:3])
print(y_test[:3])
print(predicted[:3])
print(predicted.shape) #!Bob and Eve's shape



print("Score: ", scores)
