'''
Loss Function   Categorical Cross entropy
in this we have created and array so numpy is so powerful that 
it itself pairs the row and column indices position by position
(-) because the logarithm of a probability between 0 and 1 is negative, but we want the loss to be positive.
All these values are between 0 and 1. Their natural logarithms are negative

'''

softmax_outputs = np.array([[0.7, 0.1, 0.2],
 [0.1, 0.5, 0.4],
 [0.02, 0.9, 0.08]])
class_target=[0,1,1]
print(softmax_outputs[range(len(softmax_outputs)),class_target])

loss=-np.log(softmax_outputs[range(len(softmax_outputs)),class_target])
print(loss)
average_loss=np.mean(loss)             
print(average_loss)



'''
For same cross entropy we are doing one hot encoding
Can be convenient for vectorized mathematical operations

'''
y_true = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
])
correct = np.sum(
    softmax_outputs * y_true,
    axis=1
)
loss2 = -np.log(correct)
print(loss2)


#loss calcualtion
class Loss:
    def calcu(self,output,y):
        sample_loss=self.forward(output,y)
        data_loss=np.mean(sample_loss)
        return data_loss

'''
CLipping
np.clip() keeps every value within a specified minimum and maximum range.

Here:
- 1e-7 means (0.0000001).
- 1 - 1e-7 means (0.9999999).
Consider what happens if the model assigns probability zero to the correct class:

-log(0)=+infty
The result is infinite, which can cause numerical problems during training.
Clipping prevents the logarithm from receiving zero. It also keeps probabilities away from exactly one, although the zero case is the main numerical problem here.
Important: Clipping does not make a poor prediction good. If the correct class has probability zero, its clipped probability is still tiny, so the resulting loss is extremely large.
'''
#cross entropy loss
class loss_entropy:
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clip=np.clip(y_pred,1e-7,1-1e-7)
        #if the y_true is 1-d  The shape is (3,)
        if len(y_true.shape)==1:
            #range(samples) gives [0, 1, 2], the row indices.
            #y_true gives [0, 1, 1], the column indices.
            
            correct_cof=y_pred_clip[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_cof=np.sum(y_pred_clip*y_true,axis=1)
        loss=-np.log(correct_cof)
        return loss