import numpy as np

def generate_batch(batch_size, set_size, min_val=0.0, max_val=10.0):
    X = np.random.uniform(min_val, max_val, size=(batch_size, set_size, 1))
    y = X.squeeze(-1).sum(axis=1)
    return X, y

def generate_variable_length_batch(batch_size, min_size, max_size, min_val=0.0, max_val=10.0):
    sets = []
    sums = []
    for _ in range(batch_size):
        size = np.random.randint(min_size, max_size + 1)
        s = np.random.uniform(min_val, max_val, size=(size, 1))
        sets.append(s)
        sums.append(s.sum())
    return sets, np.array(sums)

if __name__ == "__main__":
    X, y = generate_batch(batch_size=4, set_size=5)
    print("Sample set:\n", X[0].squeeze())
    print("Expected sum:", y[0])
    print("Verified sum:", X[0].sum())