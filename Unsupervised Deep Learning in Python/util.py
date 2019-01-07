from pandas import read_csv
import numpy as np
from sklearn.utils import shuffle

def get_MNIST():
	train = read_csv("train.csv").as_matrix().astype(np.float32)
	train = shuffle(train)
	
	X_train = train[:-1000, 1:]/255
	Y_train = train[:-1000, 0].astype(np.int32)
	
	X_test = train[-1000:, 1:]/255
	Y_test = train[-1000:, 0].astype(np.int32)
	
	return X_train, Y_train, X_test, Y_test

