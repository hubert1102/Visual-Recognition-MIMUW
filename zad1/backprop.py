# Lab 3
# Implement in Python + Numpy
# a Neural Network ( sigmoid( W1 * sigmoid( W0 * x + b0 ) + b1 ) )
# + Gradient Descent.
# Use 1/2 L2 as loss function.

import pickle
import numpy as np

with open('data.pkl', 'rb') as f:
  data = pickle.load(f)
training_data, test_data = data[0], data[2]

np.random.seed( 1000 )

n_input, n_hidden, n_output = len(training_data[0][0]), 100, len(training_data[0][1])
biases = [ np.random.randn(n_hidden, 1), np.random.randn(n_output, 1) ]
weights = [ np.random.randn(n_hidden, n_input), np.random.randn(n_output, n_hidden) ]

n_epochs, lr = 100, 0.00003

# TODO: implement a function which calculates the sigmoid / derivative of the sigmoid function
def sigmoid(z, deriv = False):
  return 1/(1+np.exp(-z)) if not deriv else sigmoid(z)*(1 - sigmoid(z))

# TODO: implement forward pass
def forward(x):
  wxb0 = weights[0].dot(x) + biases[0]
  hidden = sigmoid(wxb0)
  wxb1 = weights[1].dot(hidden) + biases[1]
  output = sigmoid(wxb1)
  return wxb0, hidden, wxb1, output

# TODO: implement backprop
def backprop(x, y):
  nabla_b = [ np.zeros(biases[0].shape), np.zeros(biases[1].shape) ]
  nabla_w = [ np.zeros(weights[0].shape), np.zeros(weights[1].shape) ]
  
  # forward pass
  wxb0, hidden, wxb1, output = forward( x )

  nabla_b[1] = np.subtract(output, np.reshape(y, (n_output, 1)))
  nabla_w[1] = np.matmul(nabla_b[1], hidden.T)
  nabla_b[0] = np.multiply(np.matmul(weights[1].T, nabla_b[1]), sigmoid(wxb0, True))
  nabla_w[0] = np.matmul(nabla_b[0], np.reshape(x, (n_input, 1)).T)
  return nabla_w, nabla_b

# TODO: train + evaluate
for ep in range(n_epochs):
  # train
  nabla_w = [ np.zeros(weights[0].shape), np.zeros(weights[1].shape) ]
  nabla_b = [ np.zeros(biases[0].shape), np.zeros(biases[1].shape) ]
  for x, y in training_data:
    nabla_wi, nabla_bi = backprop(x, y)
    nabla_w = [ nw + nwi for nw, nwi in zip(nabla_w, nabla_wi) ]
    nabla_b = [ nb + nbi for nb, nbi in zip(nabla_b, nabla_bi) ]
  weights = [ w - lr * nw for w, nw in zip(weights, nabla_w) ]
  biases = [ b - lr * nb for b, nb in zip(biases, nabla_b) ]
  # evaluate
  s = 0
  for x, y in test_data:
    _, _, _, output = forward( x )
    s += int(np.argmax(output) == y)
  print("Epoch {} : {} / {}".format( ep, s, len(test_data) ))

  #Epoch 99 : 8461 / 10000