import numpy as np

# Make index selection deterministic as well
np.random.seed(0)

# Matrix containing random shuffled numbers between 0 and 99
# Used to select parts of the input data 
static_index = np.arange(0, 2, dtype=np.int64)
np.random.shuffle(static_index)

# Generates a static dataset based on an operation function 
def generate_static_dataset(op_fn, num_samples=572, batch_size=5, mode='interpolation',
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

    X1_dataset = []
    X2_dataset = []

    y_dataset = []

    for i in range(batch_size):
        # Get the input stream
        X = np.random.uniform(low=0.0, high=1.00000001, size=(num_samples, 572))

        a=X[0]
        b=X[1]
        
        Y = op_fn(a, b)

        X1_dataset.append(a)
        X2_dataset.append(b)
        y_dataset.append(Y)

    return  np.array(X1_dataset), np.array(X2_dataset), np.array(y_dataset)

if __name__ == "__main__":
    generate_static_dataset(lambda x, y: x + y, 2)