import numpy as np


X = np.genfromtxt('wine_normalized.csv', delimiter=',', dtype=float)[0:140]
y = np.array(np.loadtxt('wine_normalizedOUT.csv', delimiter=","))[0:140].reshape(140,1)


class Backpropagation_Neural_Net(object):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.inputSize = 13
    self.outputSize = 1
    self.hiddenSize = 6

    self.biasIH = np.random.rand(1, self.hiddenSize) - 0.5
    self.biasHO = np.random.rand(1, self.outputSize) - 0.5

    self.WeightIH = np.random.randn(self.inputSize, self.hiddenSize) # (13x6) weight matrix from input to hidden layer
    self.WeightHO = np.random.randn(self.hiddenSize, self.outputSize) # (6x1) weight matrix from hidden to output layer
    print("WeightInputHidden:")
    print(self.WeightIH)
    print(self.WeightIH.shape)
    print("WeightHiddenOutput:")
    print(self.WeightHO)
    print(self.WeightHO.shape)
    # self.bias -= learning_rate * output_error

  def activate(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def activateDerivative(self, s):
    #derivative of activate
    return s * (1 - s)


  def forward_propagation(self, X):
    #forward_propagation through our network
    self.z = np.dot(X, self.WeightIH) + self.biasIH # dot product of X (input) and first set of 13x6 weights (+b1)
    self.z2 = self.activate(self.z) # activation function (y1)
    self.z3 = np.dot(self.z2, self.WeightHO) + self.biasHO # dot product of hidden layer (z2) and second set of 6x1 weights ()+b2
    o = self.activate(self.z3) # final activation function (y2)
    return o


  def back_propagation(self, X, y, o):
    output_delta = (y - o) * self.activateDerivative(o)
    hidden_delta = output_delta.dot(self.WeightHO.T) * self.activateDerivative(self.z2)

    self.WeightIH += X.T.dot(hidden_delta) * self.learning_rate
    self.WeightHO += self.z2.T.dot(output_delta) * self.learning_rate
    self.biasIH += self.learning_rate * np.mean(y-self.z2)
    self.biasHO += self.learning_rate * np.mean(y - o)


  def train (self, X, y):
    o = self.forward_propagation(X)
    self.back_propagation(X, y, o)

NN = Backpropagation_Neural_Net(0.2)

while np.mean(np.square(y - NN.forward_propagation(X))) > 0.00044642857:
    NN.train(X, y)

print ("Loss: \n" + str(np.mean(np.square(y - NN.forward_propagation(X))))) # mean sum squared loss
print ("\n")
good = 0
for i in range(0,30):

    print("-----------------")
    predicted = NN.forward_propagation(np.genfromtxt('wine_normalized.csv', delimiter=',', dtype=float)[141+i])
    print("Predicted Output: \n" + str(predicted))
    actual = np.genfromtxt('wine_normalizedOUT.csv', delimiter=',', dtype=float)[141+i]
    print("Actual Output: \n" + str(actual))
    if (np.abs(actual - predicted) < 0.25):
        print("Good Prediction")
        good +=1

    else:
        print("Bad Prediction")
    print("-----------------")
    print(str(good)+"/"+str(i))
    print("-----------------")



# 1/liczba el* suma kwadratow bledow
