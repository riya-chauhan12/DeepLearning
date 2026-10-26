class LayerDense:
    #input=number of input feature
    #neruon=number of neruon
    def __init__(self,n_input,n_neuron):
        #using random function to genrate randomly and the multiplying to reduce the value size
        #this generated both positive and negative value
        self.weights=0.01*np.random.randn(n_input,n_neuron)
        #this creat array of one row 3 zeros
        self.bias=np.zeros((1,n_neuron))
    def forward(self,inputs):
        
        self.output=np.dot(inputs,self.weights)+self.bias
        
class Activation_RELU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
        
class Activation_Softmax:
    def forward(self,inputs):
        exp_val=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        #softmX FUNCTION
        prob=exp_val/np.sum(exp_val,axis=1,keepdims=True)
        self.output=prob

class Loss:
    def calculate(self,output,y):
        sample_loss=self.forward(output,y)
        data_loss=np.mean(sample_loss)
        return data_loss


class Loss_entropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clip=np.clip(y_pred,1e-7,1-1e-7)
        #if the y_true is 1-d 
        if len(y_true.shape)==1:
            correct_cof=y_pred_clip[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_cof=np.sum(y_pred_clip*y_true,axis=1)
        loss=-np.log(correct_cof)
        return loss

'''
Optimizing the neural network 
so in this we are not using the gradient or backprogation
to manually understand how change in weigth and bias truly help our
model to learn. there are 2 ways we are going to approach 1. is brute forcce 2 is the bit
otimal.'''
'''
1. in this we are going to iteratate over 10000 time and
'''
import matplotlib.pyplot as plt

from nnfs.datasets import vertical_data
import nnfs
nnfs.init()
X,y=vertical_data(samples=100,classes=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()

X,y=vertical_data(samples=100,classes=3)

#create model
dense1=LayerDense(2,3)
activation1=Activation_RELU()
dense2=LayerDense(3,3)
activation2=Activation_Softmax()
loss_function=Loss_entropy()

#helper variables
lowest_loss=9999999
#copy()we want to preserve a separate snapshot
best_dense1_weights = dense1.weights.copy()
best_dense1_bias = dense1.bias.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_bias = dense2.bias.copy()

for iteration in range(100000):
    dense1.weights+=0.05*np.random.randn(2,3)
    dense1.bias+=0.05*np.random.randn(1,3)
    dense2.weights+=0.05*np.random.randn(3,3)
    dense2.bias+=0.05*np.random.randn(1,3)
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss=loss_function.calculate(activation2.output,y)
    prediction=np.argmax(activation2.output,axis=1)
    accuracy=np.mean(prediction==y)

    if(loss<lowest_loss):
        
       print('New set of weights found, iteration:',best_dense1_weights,best_dense2_weights,'weight',best_dense1_bias,best_dense2_bias,'bias',iteration,'loss:', loss, 'acc:', accuracy)
       best_dense1_weights = dense1.weights.copy()
       best_dense1_bias = dense1.bias.copy()
       best_dense2_weights = dense2.weights.copy()
       best_dense2_bias = dense2.bias.copy()
       lowest_loss=loss

'''
 I this we are only saving the best weight and bias and printing  
 which set of weight and bias minimizing our loss
 
 '''

'''
Case 2- Using ra dom search algorithm 
we find the best set of weigth and bais and 
if the loss is less then current we will update that best
else we will use the previous best weights and bias and use them 
as current weight and bias.

this only works fro simple dataset like vertical

'''


X,y=vertical_data(samples=100,classes=3)
dense1=LayerDense(2,3)
activation1=Activation_RELU()
dense2=LayerDense(3,3)
activation2=Activation_Softmax()
loss_fun=Loss_entropy()

lowest_loss=9999999
best_dense1_weights=dense1.weights.copy()
best_dense1_bias=dense1.bias.copy()
best_dense2_weights=dense2.weights.copy()
best_dense2_bias=dense2.bias.copy()

for iteration in range(10000):
    dense1.weights+=0.05*np.random.randn(2,3)
    dense1.bias+=0.05*np.random.randn(1,3)
    dense2.weights+=0.05*np.random.randn(3,3)
    dense2.bias+=0.05*np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss=loss_fun.calculate(activation2.output,y)
    pred=np.argmax(activation2.output,axis=1)
    acc=np.mean(pred==y)
    if loss<lowest_loss:
        print('New set of weights found, iteration:', iteration,'loss:', loss, 'acc:', acc)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_bias = dense1.bias.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_bias = dense2.bias.copy()
        lowest_loss=loss
    else:
        dense1.weights=best_dense1_weights.copy()
        dense1.bias=best_dense1_bias.copy()
        dense2.weights=best_dense2_weights.copy()
        dense2.bias=best_dense2_bias.copy()
    