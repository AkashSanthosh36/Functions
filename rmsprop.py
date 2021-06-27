import numpy as np

#inputs initialization
x=np.array([[1,0,1],[0,1,1],[1,1,1]])
#output initialization
y=np.array([[0],[0],[0]])
np.random.seed(0)

#defining the neural network
input_neurons=x.shape[1]#gives the number of columns
hidden_layer_neurons=2
output_neurons=1
lr=0.01#learning rate (decides the effect of change and its updation)
beta=0.9

#initialiation of weights and biases
weight0=np.random.uniform(size=(input_neurons,hidden_layer_neurons))
weight1=np.random.uniform(size=(hidden_layer_neurons,output_neurons))
bias0=np.random.uniform(size=(1,hidden_layer_neurons))
bias1=np.random.uniform(size=(1,output_neurons))
sdw0=0
sdw1=0

#sigmoid function
def sigmoid(i,deriv=False):
    if(deriv==True):
        return i*(1-i)
    else:
        a=1/(1+np.exp(-i))
        return a

for i in range(3):
    #forward propogation
    hidden_layer_input=np.dot(x,weight0)+bias0
    hidden_layer_output=sigmoid(hidden_layer_input)
    output_layer_input=np.dot(hidden_layer_output,weight1)+bias1
    output=sigmoid(output_layer_input)
    
    #backpropagation
    error_at_the_output=(y-output)

    #for output layer
    slope_output=sigmoid(output,deriv=True)
    d_output=error_at_the_output*slope_output #just multiplying the corresponding values eg:a=np.array([[1],[0],[1]]) and b=np.array([[0],[1],[1]]) ,a*b=array([[0],[0],[1]])  

    #for hidden layer
    slope_hiddenlayer=sigmoid(hidden_layer_output,deriv=True)
    error_at_hidden_layer=np.dot(d_output,weight1.T)
    d_hiddenlayer=error_at_hidden_layer*slope_hiddenlayer
    
    dweight0=np.dot(x.T,d_hiddenlayer)
    dweight1=np.dot(hidden_layer_output.T,d_output)
    print(dweight0,dweight1)
    sdw0=beta*sdw0+((1-beta)*(dweight0*dweight0))
    sdw1=beta*sdw1+((1-beta)*(dweight1*dweight1))
    print(output)
    weight0-=(lr/(sdw0**0.5))*dweight0 
    weight1-=(lr/(sdw1**0.5))*dweight1 

print(output)
a=np.array([[1,2],[3,4]])
       