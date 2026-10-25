class LayerDense:
    #input=number of input feature
    #neruon=number of neruon
    def __init__(self,n_input,n_neuron):
        #using random function to genrate randomly and the multiplying to reduce the value size
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
    def calculate(self,output,y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss



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
    dense1.weights=0.05*np.random.randn(2,3)
    dense1.bias=0.05*np.random.randn(1,3)
    dense2.weights=0.05*np.random.randn(3,3)
    dense2.bias=0.05*np.random.randn(1,3)
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss=loss_function.calculate(activation2.output,y)
    prediction=np.argmax(activation2.output,axis=1)
    accuracy=np.mean(prediction==y)

    if(loss<lowest_loss):
       best_dense1_weights = dense1.weights.copy()
       best_dense1_bias = dense1.bias.copy()
       best_dense2_weights = dense2.weights.copy()
       best_dense2_bias = dense2.bias.copy()
       lowest_loss=loss

