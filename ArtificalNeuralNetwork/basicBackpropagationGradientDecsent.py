'''
This is the optimization for single neuron
one neruon calculate z= x0w0+x1w1+x2w2 +b

then output=
RElu(x)
finally loss is calculated

L=(ouptut-target)**2

'''
#The learning rate controls how much we change the weights and biases during training.
learning_rate=0.001
def relu(x):
    return np.maximum(0,x)
'''
            1 x>0
Relu'(x)={
            0 x<=0
np.where() checks the condition:
- If x > 0, return 1.
- Otherwise, return 0.

Why do we need this? Because during backpropagation, we need to know how the ReLU output changes when its input changes.
'''
def relu_derivative(x):
    return np.where(x>0,1,0)


    
import numpy as np
weights = np.array([-3.0, -1.0, 2.0])
bias = 1.0
inputs = np.array([1.0, -2.0, 3.0])
target_output = 0.0
learning_rate = 0.001

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

for iteration in range(200):
    #forward pass
    linear_output=np.dot(inputs,weights)+bias
    output=relu(linear_output)
    loss = (output - target_output) ** 2

    #backward pass

    #Derivative 1: Loss with respect to output(relu)
    dloss_doutput=2*(output-target_output)
    
    #Derivative 2: Output with respect to linear output
    doutput_dlinear=relu_derivative(linear_output)

    '''
    
    #Derivative 3: Linear output with respect to the multiplication term
     dz/d(x0,w0) z is sum here

     When differentiating with respect to the complete term \(x_0w_0\), its derivative is 1.
     Derivative 4: Multiplication term with respect to weight
     d(x0w0)/dw0=x0 value of x0 in the given code is in inputs
     Because \(x_0\) is a constant and we differentiate with respect to \(w_0\).
     
     
     
     
    '''
    dlinear_dweights=inputs
    
    dlinear_dbias=1.0
#. Now multiply all the derivatives
    dloss_dlinear = dloss_doutput * doutput_dlinear
    dloss_dweights = dloss_dlinear * dlinear_dweights
    dloss_dbias = dloss_dlinear * dlinear_dbias

    #update weights
    weights-=learning_rate*dloss_dweights
    bias-=learning_rate*dloss_dbias
    # Print the loss for this iteration
    print(f"Iteration {iteration + 1}, Loss: {loss}")
print("Final weights:", weights)
print("Final bias:", bias)


    
