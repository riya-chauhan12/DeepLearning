'''
Basic neuron with 3 inputs
x1 
     \w1
    w2 \
x2  -- O --> x1*w1+x2*w2+x3*w3 +bias  
   w3 /|
    /  |
x3      bias

'''


input=[1,2,3]
weights=[0.2,0.8,-0.5]
bias=2
outputs=(input[0]*weights[0]+input[1]*weights[1]+input[2]*weights[2]+bias)
print(outputs)


'''
Layers of neuron  
4 inputs 3 neurons(all 4 input connect to every one of neruon) +bias for each neuron 
so every neuron will have four weights comming to it
'''
inp=[1,2,3,2.5]
weights=[[0.2,0.8,-0.5,1],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]
w1=weights[0] # list of weight assosiated with 1st neuron =w11,w12,w13,w14
w2=weights[1]
w3=weights[2]
bias=[2,3,0.5]
b1=2
b2=3
b3=0.5
output=[
    inp[0]*w1[0]+inp[1]*w1[1] +inp[2]*w1[2]+inp[3]*w1[3]+b1 ,
     inp[0]*w2[0]+inp[1]*w2[1] +inp[2]*w2[2]+inp[3]*w2[3]+b2,
 inp[0]*w3[0]+inp[1]*w3[1] +inp[2]*w3[2]+inp[3]*w3[3]+b3
]
print(output)

#using loops

inp=[1,2,3,2.5]
weights=[[0.2,0.8,-0.5,1],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]
bias=[2,3,0.5]
layer_output=[]
#for each neuron
for n_weig,n_b in  zip(weights,bias):
    n_output=0
    #mutiply input with weight
    for n_inp,weight in zip(inp,n_weig):
        n_output+=n_inp*weight
    #add bias
    n_output+=n_b
    #append in final ouptut
    layer_output.append(n_output)
print(layer_output)



#layers of neuron using numpy
import numpy as np
inp=[1,2,3,2.5]
weights=[[0.2,0.8,-0.5,1],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]
bias=[2,3,0.5]
multi=np.dot(weights,inp)
print(multi)
output=np.add(multi,bias)
print(output)
        