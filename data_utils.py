import numpy as np

# Make index selection deterministic as well
np.random.seed(0)

# Matrix containing random shuffled numbers between 0 and 99
# Used to select parts of the input data 
static_index = np.arange(0, 2, dtype=np.int64)
np.random.shuffle(static_index)

# Generates a static dataset based on an operation function 
def generate_static_dataset(op_fn, num_samples=864, mode='interpolation',
                            seed=0):
    """
    Generates a dataset given an operation and a mode of working.
    Used to generate the synthetic static dataset.

    # Arguments:
        op_fn: A function which accepts 2 numpy arrays as arguments
            and returns a single numpy array as the result.
        num_samples: Number of samples for the dataset.
        mode: Can be one of `interpolation` or `extrapolation`
        seed: random seed

    Returns:

    """
    assert mode in ['interpolation', 'extrapolation']
    assert callable(op_fn)

    np.random.seed(seed)  # make deterministic

    print("Generating dataset")

    X_dataset = []
    y_dataset = []

    for i in range(num_samples):

        # Get the input stream
        X = np.random.rand(num_samples, 2)

        if mode == 'extrapolation':
            X *= 100.

        # Select the slices on which we will perform the operation
        a_index, b_index = static_index[:(len(static_index) // 2)], static_index[(len(static_index) // 2):]
        a = X[:, a_index]
        b = X[:, b_index]

        # Get the sum of the slices
        a = np.sum(a, axis=-1, keepdims=True)
        b = np.sum(b, axis=-1, keepdims=True)

        # perform the operation on the slices in order to get the target
        Y = op_fn(a, b)
        X_dataset.append(X)
        y_dataset.append(Y)

    return np.array(X_dataset), np.array(y_dataset)
