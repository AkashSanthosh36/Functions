
#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np


# In[10]:


#inputs initialization
x=np.array([[1],[0],[1]])

#output initialization
y=np.array([[1],[0],[1]])
np.random.seed(0)


# In[11]:


#defining the neural network
input_neurons=x.shape[1]#gives the number of columns
hidden_layer_neurons=2
output_neurons=1
lr=0.9#learning rate (decides the effect of change and its updation)


# In[12]:


#initialiation of weights and biases
weight0=np.random.uniform(size=(input_neurons,hidden_layer_neurons))
weight1=np.random.uniform(size=(hidden_layer_neurons,output_neurons))
bias0=np.random.uniform(size=(1,hidden_layer_neurons))
bias1=np.random.uniform(size=(1,output_neurons))


# In[13]:


#sigmoid function
def sigmoid(i,deriv=False):
    if(deriv==True):
        return i*(1-i)
    else:
        a=1/(1+np.exp(-i))
        return a


# In[14]:


#Relu function
def relu(i,deriv=False):
    if(deriv==True):
        return 1
    else:
        return i


# In[15]:


#hyperbolic tangent function
def tanh(i,deriv=False):
    a=np.exp(i)
    b=np.exp(-i)
    c=((a-b)/(a+b))
    if(deriv==True):
        return (1-(c**2))
    else:
        return c


# In[16]:


for i in range(10000):
    #forward propogation
    hidden_layer_input=np.dot(x,weight0)+bias0
    hidden_layer_output=tanh(hidden_layer_input)
    output_layer_input=np.dot(hidden_layer_output,weight1)+bias1
    output=tanh(output_layer_input)

    #backpropagation
    error_at_the_output=(y-output)

    #for output layer
    slope_output=tanh(output,deriv=True)
    d_output=error_at_the_output*slope_output #just multiplying the corresponding values eg:a=np.array([[1],[0],[1]]) and b=np.array([[0],[1],[1]]) ,a*b=array([[0],[0],[1]])  


    #for hidden layer
    slope_hiddenlayer=tanh(hidden_layer_output,deriv=True)
    error_at_hidden_layer=np.dot(d_output,weight1.T)
    d_hiddenlayer=error_at_hidden_layer*slope_hiddenlayer

    #updation of weights
    weight0=weight0+(np.dot(x.T,d_hiddenlayer)*lr)
    weight1=weight1+(np.dot(hidden_layer_output.T,d_output)*lr)

print(output)


# In[ ]:




