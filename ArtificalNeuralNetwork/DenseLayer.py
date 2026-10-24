''' 
this generates the spiral dataset for training the neural 
nnfs =neuralnetwork from scratch

'''
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import nnfs
#consist of setting 
nnfs.init()
#datset will be  generate 100 poitn for each class 100* 3 class=300
#x shape(300,2) y shape(300,)
X,y=spiral_data(samples=100,classes=3)
#X contain input and y contain target
#create plot of first feature on x-axis and the second feataure on y-axis
plt.scatter(X[:,0],X[:,1])
plt.show()
# to see the 3 classes clearly with color
plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()


'''
Dense fully connected layer  and passing spiral dataset to it
for that install nnfs library

'''
import numpy as np
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()

class LayerDense:
    #input=number of input feature
    #neruon=number of neruon
    def __init__(self,n_input,n_neuron):
        #using random function to genrate randomly and the multiplying to reduce the value size
        self.weight=0.01*np.random.randn(n_input,n_neuron)
        #this creat array of one row 3 zeros
        self.bias=np.zeros((1,n_neuron))
    def forward(self,inputs):
        
        self.output=np.dot(inputs,self.weight)+self.bias
X,y=spiral_data(samples=300,classes=3)
# we are intializing with 2 input and 3 neruon
dense1=LayerDense(2,3)
# passing X which is feature (300,2) 
# we are passin 300 sample with 2 feature
# this will method calculate oupt dot((300,2),(2,3)) +(1,3)=(300,3)
dense1.forward(X)
print(dense1.output[:5])
print(dense1.weight.shape)