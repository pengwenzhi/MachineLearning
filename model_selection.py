import numpy as np
def train_test_split(X,Y,test_ratio=0.2,seed=None):
    assert X.shape[0]==y.shape[0],\
        "the size of X must be equal to the size of Y"
    assert 0.0<=test_ratio<=1.0,\
        "test_ratio must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes=np.random.premutation(len(X))
    test_size=int(len(X)*test_ratio)
    test_indexes=shuffled_indexes[:test_size]
    train_indexes=shuffled_indexes[test_size:]

    X_train=X[train_indexes]
    Y_train=Y[train_indexes]

    X_test=X[test_indexes]
    Y_test=Y[test_indexes]