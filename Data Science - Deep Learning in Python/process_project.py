from pandas import read_csv
from numpy import zeros

def get_data(data_file_path):
	df = read_csv(data_file_path)
	data = df.as_matrix()

	X = data[:, :-1]
	Y = data[:, -1]
	lis = [1,2]
	for i in lis:
		X[:,lis] = (X[:,lis] - X[:,lis].mean()) / X[:,lis].std()

	N, D = X.shape

	X_final = zeros((N, D+3))
	X_final[:, 0:(D-1)] = X[:, 0:(D-1)]
	for i in range(N):
		t = int(X[i, D-1])
		X_final[i, t+D-1] = t

	return X_final, Y

def get_binary_data(data_file_path):
	X, Y = get_data(data_file_path)
	X2 = X[Y <= 1]
	Y2 = Y[Y <= 1]
	return X2, Y2

