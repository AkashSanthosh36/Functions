#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np


# In[42]:


#inputs initialization
x=np.array([[1,0,1],[0,1,1],[1,1,1]])
#output initialization
y=np.array([[0],[0],[1]])
np.random.seed(0)


# In[43]:


#defining the neural network
input_neurons=x.shape[1]#gives the number of columns
hidden_layer_neurons=2
output_neurons=1
lr=0.1#learning rate (decides the effect of change and its updation)
beta=0.9


# In[44]:


#initialiation of weights and biases
weight0=np.random.uniform(size=(input_neurons,hidden_layer_neurons))
weight1=np.random.uniform(size=(hidden_layer_neurons,output_neurons))
bias0=np.random.uniform(size=(1,hidden_layer_neurons))
bias1=np.random.uniform(size=(1,output_neurons))
vdw0=0
vdw1=0


# In[45]:


#sigmoid function
def sigmoid(i,deriv=False):
    if(deriv==True):
        return i*(1-i)
    else:
        a=1/(1+np.exp(-i))
        return a


# In[46]:


#Relu function
def relu(i,deriv=False):
    if(deriv==True):
        return 1
    else:
        return i


# In[47]:


#hyperbolic tangent function
def tanh(i,deriv=False):
    a=np.exp(i)
    b=np.exp(-i)
    c=((a-b)/(a+b))
    if(deriv==True):
        return (1-(c**2))
    else:
        return c


# In[48]:


for i in range(1000):
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
    
    #previous values of v
    p_vdw0=vdw0
    p_vdw1=vdw1
    vdw0=beta*vdw0+((1-beta)*dweight0)
    vdw1=beta*vdw1+((1-beta)*dweight1)
    
    #weight updation
    weight0=weight0-(beta*p_vdw0)+(1+beta)*vdw0
    weight1=weight1-(beta*p_vdw1)+(1+beta)*vdw1
    
print(output)


# In[ ]:




