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



  '''
for network of 3 neruon and 4 inputs we have
4 inputs

   |

3 neurons
    |

4 weights for each neuron
    |

3 biases
    |
actiavtion
    |
sum of all activation
    |
    loss function



1 final output, created by adding the three neuron activations


'''
import numpy as np
inputs=np.array([1,2,3,4])

weights=np.array([
    [ 0.1,0.2,0.3,0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
])

biases=np.array([0.1,0.2,0.3])
#The learning rate controls how much we change the weights and biases during training.
learning_rate=0.001


for iteration in range(200):
    #forward pass
    z=np.dot(weights,inputs)+biases
    #We pass the weighted sums through ReLU.
    a=relu(z)
    #We add all three neuron outputs:
    #a=[3.1,7.2,11.3]
    #We add all three neuron outputs: y
    y=np.sum(a)

    #calculating loss
    loss=y**2

    ###BACKWARD PASS####
    #gradient loss with respect to y(relu)
    ##Our loss fucntion is y=x**2 which has a derviative 2x 
    #Derivative 1: Loss with respect to output
    dL_dy=2*y
    #Derivative 2: Output with respect to activations
    #becuase in this we have 3 neuron each one  has its own activation function
    dy_da = np.ones_like(a) #dy_da = np.ones_like(a) 

    #Now multiply the first two derivatives:
    dL_da = dL_dy * dy_da
    
    #Derivative 3: Activation with respect to weighted sum
    da_dz = relu_derivative(z)

    dL_dz = dL_da * da_dz

    #Derivatives 4 and 5: Weighted sum with respect to weights
    #np.outer= It takes two arrays and multiplies every value of the first array with every value of the second array.
    dL_dW = np.outer(dL_dz, inputs)
    
    #Bias derivative
    dL_db = dL_dz
    # Update weights and biases
    weights -= learning_rate * dL_dW
    biases -= learning_rate * dL_db
     # Print the loss every 20 iterations
    if iteration % 20 == 0:
        print(f"Iteration {iteration}, Loss: {loss}")


# Final weights and biases
print("Final weights:\n", weights)
print("Final biases:\n", biases)


      
