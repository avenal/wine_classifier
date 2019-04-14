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
