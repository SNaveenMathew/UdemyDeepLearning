from keras.models import Sequential
from keras.layers import Dense
from numpy.random import seed
from pandas import read_csv

seed(1)
dataframe = read_csv("BBC.csv") # Variables not scaled
# It is better to scale or normalize variables as neural networks are susceptible to variable scales
dataframe = read_csv("BBCN.csv") # Variables normalized to similar scales
array = dataframe.values
X = array[:, 0:11]
Y = array[:, 11]

# Feed forward neural network
model = Sequential()
# Input layer has 11 neurons. Dense => fully connected
model.add(Dense(11, input_dim = 11, init = "uniform", activation = 'relu'))
# First hidden layer has 8 neurons. Dense => fully connected
model.add(Dense(8, init = "uniform", activation = 'relu'))
# Second hidden layer has 8 neurons. Dense => fully connected
model.add(Dense(8, init = "uniform", activation = 'relu'))
# Output layer has 11 neurons. Dense => fully connected
model.add(Dense(1, init = "uniform", activation = 'sigmoid'))

# Compiling the model
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
# Fitting the model
model.fit(X, Y, nb_epoch = 20, batch_size = 10)
# Scoring the model
scores = model.evaluate(X, Y)
print(str(model.metrics[0]) + ": " + str(scores[0]*100))

# Tuning the model: Number of layers
model = Sequential()
model.add(Dense(11, input_dim = 11, init = "uniform", activation = 'relu'))
model.add(Dense(8, init = "uniform", activation = 'relu'))
model.add(Dense(8, init = "uniform", activation = 'relu'))
model.add(Dense(8, init = "uniform", activation = 'relu'))
model.add(Dense(1, init = "uniform", activation = 'sigmoid'))
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X, Y, nb_epoch = 20, batch_size = 10)
scores = model.evaluate(X, Y)
print(str(model.metrics[0]) + ": " + str(scores[0]*100))

# Tuning the model: Number of neurons
model = Sequential()
model.add(Dense(11, input_dim = 11, init = "uniform", activation = 'relu'))
model.add(Dense(4, init = "uniform", activation = 'relu'))
model.add(Dense(4, init = "uniform", activation = 'relu'))
model.add(Dense(1, init = "uniform", activation = 'sigmoid'))
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X, Y, nb_epoch = 20, batch_size = 10)
scores = model.evaluate(X, Y)
print(str(model.metrics[0]) + ": " + str(scores[0]*100))

# Tuning the model: Number of epochs part 1
model = Sequential()
model.add(Dense(11, input_dim = 11, init = "uniform", activation = 'relu'))
model.add(Dense(8, init = "uniform", activation = 'relu'))
model.add(Dense(8, init = "uniform", activation = 'relu'))
model.add(Dense(1, init = "uniform", activation = 'sigmoid'))
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X, Y, nb_epoch = 200, batch_size = 10)
scores = model.evaluate(X, Y)
print(str(model.metrics[0]) + ": " + str(scores[0]*100))

# Tuning the model: Number of epochs part 2
model = Sequential()
model.add(Dense(11, input_dim = 11, init = "uniform", activation = 'relu'))
model.add(Dense(8, init = "uniform", activation = 'relu'))
model.add(Dense(8, init = "uniform", activation = 'relu'))
model.add(Dense(1, init = "uniform", activation = 'sigmoid'))
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X, Y, nb_epoch = 500, batch_size = 10)
scores = model.evaluate(X, Y)
print(str(model.metrics[0]) + ": " + str(scores[0]*100))
