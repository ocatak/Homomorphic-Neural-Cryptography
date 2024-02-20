from networks import HO_model, c3_bits
from data_utils import generate_static_dataset

task_fn = lambda x, y: x + y
num_samples = c3_bits
batch_size = 512

weights_path = 'weights/addition_weights.h5'

HO_model.load_weights(weights_path)


X1, X2, y = generate_static_dataset(task_fn, num_samples, batch_size)

predicted = HO_model.predict([X1, X2], 128)

print(predicted)
print(y)