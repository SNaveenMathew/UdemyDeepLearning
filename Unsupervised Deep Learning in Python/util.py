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

def get_donut_data():
	N = 500
	rad_inner = 10
	rad_outer = 20
	r1 = rad_inner + np.random.randn(int(N/2))
	theta = 2 * np.pi * np.random.random(int(N/2))
	x_inner = np.concatenate([[r1 * np.cos(theta)], [r1 * np.sin(theta)]]).T
	r2 = rad_outer + np.random.randn(int(N/2))
	theta = 2 * np.pi * np.random.random(int(N/2))
	x_outer = np.concatenate([[r2 * np.cos(theta)], [r2 * np.sin(theta)]]).T
	
	X = np.concatenate([x_inner, x_outer])
	Y = np.array([0] * int(N/2) + [1] * int(N/2))
	return X, Y

def get_xor_data():
	x1 = np.random.random((100, 2))
	x2 = np.random.random((100, 2)) - np.array([1, 1])
	x3 = np.random.random((100, 2)) - np.array([1, 0])
	x4 = np.random.random((100, 2)) - np.array([0, 1])
	x = np.vstack([x1, x2, x3, x4])
	y = np.array([0] * 200 + [1] * 200)
	return x, y

def get_gauss_cloud(pts_per_cloud = 50):
	centers = []
	x = [0, 2]
	for i in x:
		for j in x:
			for k in x:
				centers.append([i, j, k])

	centers = np.array(centers)
	dat = []

	for ctr in centers:
		cloud = np.random.randn(pts_per_cloud, 3) * 0.5 + ctr
		dat.append(cloud)
	
	dat = np.concatenate(dat)
	return centers, dat

def error_rate(pred, trgt):
    return np.mean(pred != trgt)

