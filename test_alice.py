from networks import alice, p1_bits, p2_bits
import numpy as np
from EllipticCurve import generate_key_pair
from keras.models import Model
from keras.layers import Input, Reshape
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
# This folder store the trained model weights
if not os.path.exists('weights'):
    os.makedirs('weights')


# Hyper parameters used for training
units = 2
batch_size = 5
num_samples = 864
# num_samples = 864*batch_size #rows

# Create a task function for addition
task_name = 'addition'
task_fn = lambda x, y: x + y

# Generate the model with an input layer and two NAC layers
ip = Input(shape=(864,2,))
x = NAC(units)(ip)
x = NAC(1)(x)
x = Reshape((864,))(x)

model = Model(ip, x)

model_out = model([ip])
print(model_out)

# Compile the model
# Use RMSprop as the optimizer and mean squared error as the loss function
optimizer = RMSprop(0.1)
model.compile(optimizer, 'mse')

# Generate training and testing datasets
X_train, y_train = generate_static_dataset(task_fn, num_samples, mode='interpolation')

private_arr, public_arr = generate_key_pair(batch_size)

p1_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
p2_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
cipher1, cipher2 = alice.predict([public_arr, p1_batch, p2_batch])

X_test = []
y_test = []
for i in range(len(cipher1)):
    X_bit_array = []
    y_bit_array = []
    for j in range(len(cipher1[i])):
        X_bit_array.append([cipher1[i][j], cipher2[i][j]])
        y_bit_array.append([cipher1[i][j] + cipher2[i][j]])
    X_test.append(X_bit_array)
    y_test.append(y_bit_array)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_test.shape)


# X_test, y_test = generate_static_dataset(task_fn, num_samples, mode='extrapolation')

# Use 'ModelCheckpoint' callback to save the model weights during training, specifically saving only the best-performing weights based on validation loss
weights_path = 'weights/%s_weights.h5' % (task_name)
checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                             verbose=1, save_weights_only=True, save_best_only=True)

callbacks = [checkpoint]

# Train model and includes validation using the test dataset
model.fit(X_train, y_train, batch_size=64, epochs=500,
          verbose=2, callbacks=callbacks, validation_data=(X_test, y_test))

# Evaluate the model on the test data set and the mean squared error of the model on the test dataset is printed
model.load_weights(weights_path)

scores = model.evaluate(X_test, y_test, batch_size=128)
predicted = model.predict(X_test, 128)
print(X_test[:3])
print(y_test[:3])
print(predicted[:3])
print(predicted.shape) #!Bob and Eve's shape



print("Score: ", scores)
