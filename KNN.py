import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:
    def __init__(self,k):
        assert k >=1,"k must be valid"
        self.k=k
        self._X_train=None
        self._Y_train=None
    def fit(self,X_train,Y_train):
        assert X_train.shape[0] == Y_train.shape[0], \
            "the size of X_train must equal to the size of Y_train"
        assert self.k<=X_train.shape[0],\
            "the size of X_train must be at least k."

        self._X_train=X_train
        self._Y_train=Y_train
        return self
    def predict(self,X_predict):
        assert self._X_train is not None and self._X_train is not None,\
            "must fit before predict!"
        assert X_predict.shape[1]==self._X_train.shape[1],\
            "the feature number of X_predict must be equal to X_train"
        y_predict =[self._predict(x) for x in X_predict]
        return np.array(y_predict)
    def _predict(self,x):
        assert x.shape[0]==self._X_train.shape[1],\
            "the feature number of x must be equal to X_train"
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_Y = [self._Y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_Y)
        return votes.most_common(1)[0][0]
    def __repr__(self):
        return "KNN(k=%d)"%self.k


def KNN_classify(k,X_train,Y_train,x):
    assert 1<=k<=X_train.shape[0],"K must be vaild"
    assert X_train.shape[0]==Y_train.shape[0],\
        "the size of X_train must equal to the size of Y_train"
    assert X_train.shape[1]==x.shape[0],\
        "the feature number of x must be equal to X_train"
    distances=[sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
    nearest=np.argsort(distances)
    topK_Y=[Y_train[i] for i in nearest[:k]]
    votes=Counter(topK_Y)
    return votes.most_common(1)[0][0]