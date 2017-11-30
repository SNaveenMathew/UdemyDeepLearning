from pandas import read_csv
from numpy import zeros, argmax
from numpy.random import randn
from softmax import softmax
from forwardpass import forwardpass, accuracy
from process_project import get_data

X, Y = get_data("data/ecommerce_data.csv")

# Number of hidden units in hidden layer
M = 5
D = X.shape[1]
K = len(set(Y))

W1 = randn(D, M)
b1 = zeros(M)
W2 = randn(M, K)
b2 = zeros(K)

p_Y_given_X = forwardpass(X, W1, b1, W2, b2, activation = "tanh")
predictions = argmax(p_Y_given_X, axis = 1)
print(accuracy(Y, predictions))
