from pandas import read_csv
from numpy import zeros, argmax
from numpy.random import randn
from softmax import softmax
from forwardpass import forwardpass, accuracy
from process_project import get_data
import matplotlib.pyplot as plt
from cost_derivative import cost, derivative_w2, derivative_b2, derivative_w1, derivative_b1

X, Y = get_data("data/ecommerce_data.csv")

# Number of hidden units in hidden layer
M = 5
D = X.shape[1]
K = len(set(Y))

W1 = randn(D, M)
b1 = zeros(M)
W2 = randn(M, K)
b2 = zeros(K)

alpha = 10e-7
costs = []

for epochs in range(10e5):
    output, hidden = forwardpass(X, W1, b1, W2, b2)
    if epochs % 100 == 0:
        c = cost(T, output)
        predictions = argmax(output, axis = 1)
        costs.append(c)
        print(accuracy(Y, predictions))

    dc_dw2 = derivative_w2(hidden, Y, output)
    dc_db2 = derivative_b2(Y, output)
    dc_dw1 = derivative_w1(X, hidden, Y, output, W2)
    dc_db1 = derivative_b1(Y, output, W2, hidden)
    W2 += learning_rate * dc_dw2
    b2 += learning_rate * dc_db2
    W1 += learning_rate * dc_dw1
    b1 += learning_rate * dc_db1

plt.plot(costs)
plt.show()
