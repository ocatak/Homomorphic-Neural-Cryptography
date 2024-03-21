from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv1D, concatenate, Lambda, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from key.EllipticCurve import get_key_shape, set_curve
from neural_network.nalu import NALU
from data_utils.dataset_generator import generate_static_dataset, generate_cipher_dataset
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-op', type=str, default="Adam", help='Optimizer')
parser.add_argument('-lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('-e', type=int, default=1500, help='Number of epochs')
parser.add_argument('-b', type=int, default=256, help='Batch size')
args = parser.parse_args()

c1_bits = 16
c2_bits = 16
c3_bits = 16
task_a = lambda x, y: x + y
task_m = lambda x, y: x * y
batch_size = 512

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

if args.op == "RMS":
    optimizer = RMSprop(args.lr)
else:
    optimizer = Adam(args.lr)

HO_model.compile(optimizer, 'mse')

HO_model.trainable = True

X1_train_a, X2_train_a, y_train_a = generate_static_dataset(task_a, c3_bits, batch_size)
X1_test_a, X2_test_a, y_test_a = generate_static_dataset(task_a, c3_bits, batch_size)
op_a = np.zeros(X1_train_a.shape)

X1_train_m, X2_train_m, y_train_m = generate_static_dataset(task_m, c3_bits, batch_size)
X1_test_m, X2_test_m, y_test_m = generate_static_dataset(task_m, c3_bits, batch_size)
op_m = np.ones(X1_train_m.shape)

X1_train = np.concatenate((X1_train_a, X1_train_m))
X2_train = np.concatenate((X2_train_a, X2_train_m))
y_train = np.concatenate((y_train_a, y_train_m))
X1_test = np.concatenate((X1_test_a, X1_test_m))
X2_test = np.concatenate((X2_test_a, X2_test_m))
y_test = np.concatenate((y_test_a, y_test_m))
operation = np.concatenate((op_a, op_m))

HO_model.fit([operation, X1_train, X2_train], y_train, batch_size=args.b, epochs=args.e,
    verbose=2, validation_data=([operation, X1_test, X2_test], y_test))

HO_model.trainable = False

predicted_a = HO_model.predict([op_a, X1_test_a, X2_test_a], 128)
print(X1_test_a[:1])
print(X2_test_a[:1])
print(y_test_a[:1])
print(predicted_a[:1])
print()
predicted_m = HO_model.predict([op_m, X1_test_m, X2_test_m], 128)
print(X1_test_m[:1])
print(X2_test_m[:1])
print(y_test_m[:1])
print(predicted_m[:1])

tolerance = 1e-4
correct_elements = np.sum(np.abs(y_test_a - predicted_a) <= tolerance)
total_elements = np.prod(predicted_a.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Addition: {accuracy_percentage:.2f}%")

tolerance = 1e-4
correct_elements = np.sum(np.abs(y_test_m - predicted_m) <= tolerance)
total_elements = np.prod(predicted_m.shape)
accuracy_percentage = (correct_elements / total_elements) * 100
print(f"HO model Accuracy Percentage Multiplication: {accuracy_percentage:.2f}%")

print(f"epoch {args.e}, batch {args.b}, optimizer {args.op}, learning rate {args.lr}")