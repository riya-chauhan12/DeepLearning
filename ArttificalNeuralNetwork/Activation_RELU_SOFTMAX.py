'''
Applying a RELU in a dense layer
np.maximum() performs an element-by-element comparison when given an array or list.

'''
import numpy as np

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
#f(x)=max(0,x)
ouput=np.maximum(0,inputs)
print(ouput)

'''
Creating RELU activation function '''
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import nnfs


class Activation_RELU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
#passing data through layer
X,y=spiral_data(samples=100,classes=3)
dense1=LayerDense(2,3)
activation1=Activation_RELU()
dense1.forward(X)
activation1.forward(dense1.output)
print(activation1.output[:5])

'''
Activation using softmax
np.exp() is a NumPy function that calculates the exponential of a number or every element in an array.
'''
inputs = [
    [1, 2, 3, 2.5],
    [2., 5., -1., 2],
    [-1.5, 2.7, 3.3, -0.8]
]
#for normalzation
exp_val=np.exp(inputs-np.max(inputs,axis=0,keepdims=True))
#softmX FUNCTION
prob=np.exp(exp_val/np.sum(exp_val,axis=1,keepdims=True))
print(prob)
'''

Output
[0.  2.  0.  3.3 0.  1.1 2.2 0. ]

[[0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [3.19098799e-05 0.00000000e+00 0.00000000e+00]
 [4.97558503e-05 0.00000000e+00 0.00000000e+00]
 [5.31514752e-06 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 1.09269895e-04 2.17802262e-05]]
'''

'''creating activation softmax class 
'''
class Activation_Softmax:
    def forward(self,inputs):
        exp_val=np.exp(inputs-np.max(inputs,axis=0,keepdims=True))
        #softmX FUNCTION
        prob=np.exp(exp_val/np.sum(exp_val,axis=1,keepdims=True))
        self.output=prob
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_RELU()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = LayerDense(3, 3)
# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)
dense2 = LayerDense(3, 3)
# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)
# Let's see output of the first few samples:
print(activation2.output[:5])

'''
Output
[[1.18581953 1.02333382 1.40946777 1.58929094]
 [1.46472123 1.46472123 1.00519208 1.26047859]
 [1.02620605 1.08968278 2.35525474 1.03210037]]


[[1.3956811 1.3956027 1.3955536]
 [1.3956811 1.3956027 1.3955536]
 [1.3956811 1.395603  1.3955531]
 [1.3956811 1.3956031 1.3955531]
 [1.3956811 1.3956028 1.3955532]]
'''
