from numpy.random import randn
from numpy import exp, argmax, tanh
import matplotlib.pyplot as plt
from softmax import softmax

def forwardpass(X, W1, b1, W2, b2, activation = "sigmoid"):
	if(activation == "sigmoid"):
		z1 = 1 / (1 + exp(-X.dot(W1) - b1))
	else:
		z1 = tanh(X.dot(W1) + b1)
	z2 = z1.dot(W2) + b2
	ret = softmax(z2)
	return ret, z1


def accuracy(Y, P):
	i = 0
	correct = 0
	while i<len(Y):
		if(P[i]==Y[i]):
			correct += 1
		i += 1

	return correct / len(Y)

def do_forward_pass(X, W1, b1, W2, b2, Y):
	probs = forwardpass(X, W1, b1, W2, b2)
	P = argmax(probs, axis = 1)
	acc = accuracy(Y, P)
