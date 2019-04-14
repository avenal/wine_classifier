import numpy as np


X = np.genfromtxt('data_processedIN.csv', delimiter=',', dtype=float)[0:140]
y = np.array(np.loadtxt('data_processedOUT.csv', delimiter=","))[0:140].reshape(140,1)


class Backpropagation_Neural_Net(object):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.inputSize = 13
    self.outputSize = 1
    self.hiddenSize = 6


    self.WeightIH = np.random.randn(self.inputSize, self.hiddenSize) # (13x6) weight matrix from input to hidden layer
    self.WeightHO = np.random.randn(self.hiddenSize, self.outputSize) # (6x1) weight matrix from hidden to output layer
    print("WeightInputHidden:")
    print(self.WeightIH)
    print(self.WeightIH.shape)
    print("WeightHiddenOutput:")
    print(self.WeightHO)
    print(self.WeightHO.shape)

  def activate(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def activateDerivative(self, s):
    #derivative of activate
    return s * (1 - s)


  def forward_propagation(self, X):
    #forward_propagation through our network
    self.z = np.dot(X, self.WeightIH) # dot product of X (input) and first set of 13x6 weights
    self.z2 = self.activate(self.z) # activation function
    self.z3 = np.dot(self.z2, self.WeightHO) # dot product of hidden layer (z2) and second set of 6x1 weights
    o = self.activate(self.z3) # final activation function
    return o


  def back_propagation(self, X, y, o):
    output_delta = (y - o) * self.activateDerivative(o)
    hidden_delta = output_delta.dot(self.WeightHO.T) * self.activateDerivative(self.z2)

    self.WeightIH += X.T.dot(hidden_delta) * self.learning_rate
    self.WeightHO += self.z2.T.dot(output_delta) * self.learning_rate

  def train (self, X, y):
    o = self.forward_propagation(X)
    self.back_propagation(X, y, o)

NN = Backpropagation_Neural_Net(0.2)

while np.mean(np.square(y - NN.forward_propagation(X))) > 0.01:
    NN.train(X, y)

print ("Loss: \n" + str(np.mean(np.square(y - NN.forward_propagation(X))))) # mean sum squared loss
print ("\n")

for i in range(0,30):
    print ("Predicted Output: \n" + str(NN.forward_propagation(np.genfromtxt('data_processedIN.csv', delimiter=',', dtype=float)[141+i])))
    print ("Actual Output: \n" + str(np.genfromtxt('data_processedOUT.csv', delimiter=',', dtype=float)[141+i]))
