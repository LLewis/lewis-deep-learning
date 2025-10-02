import numpy as np
import torch
import torch.nn as nn  #contains multiple activation (non-linear) functions, ReLU,Sigmoid,also Softmax, MSE
import matplotlib.pyplot as plt
from IPython import display
display.display_svg()

'''
Professor Lewis
Linear Regression model, that will predict the value of y
based on the values of x
The model will use gradient descent to train the parameters (weights) in model
The model will use Mean Squared Error MSE, to generate losses (errors) to teach the model, what to adjust for the next epoch
y hat - is the final output and will compare its predicted value against y the real data
x sub 1 refers to the values along the x axis
'''
#1. creat data
N = 30
x = torch.randn(N,1) # x values are random numbers
y = x + torch.randn(N,1)/2   # y value is equal to x plus more random numbers that are scaled down with /2

# and plot
plt.plot(x,y, 's')  # a linear relationship between x and y
#plt.show()

# 2. build model using Pytorch
ANNreg = nn.Sequential(
    nn.Linear(1,1), #input layer unit x value
    nn.ReLU(),  #activation function
    nn.Linear(1,1) #output layer unit only 1 output because we're predicting just y
)

# print(ANNreg)
#output
'''
Sequential(
(0): Linear(in_features=1, out_features=1, bias=True)
(1): ReLU()
(2): Linear(in_features=1, out_features=1, bias=True)
)
'''
#3. Learning rate
learningRate = .05

#4. Loss Function
lossfun = nn.MSELoss()

#5. optimizer (the flavor of stochastic gradient descent to implement)
optimizer = torch.optim.SGD(ANNreg.parameters(),lr=learningRate)

#6. train the model using numnber of iterations
numepochs = 200  #was 500
losses = torch.zeros(numepochs)

##Train the model
for epochi in range(numepochs):

    #forward pass
    yHat = ANNreg(x)  # using observed data  x random values - pass into deep learning model - returns the prediction yHat - final output

    #compute loss
    loss = lossfun(yHat, y)  #predicted values and real values - returns the loss -  squared data point difference of predicted data point
    losses[epochi] = loss    # viaualize losses over training

    #backprop
    optimizer.zero_grad() # re-initialize the gradient set all derivatives back to zero
    loss.backward()  # implements backprop based on losses line 62
    optimizer.step()



    #*********
    #can also manually compute losses
    #final forward pass
    predictions = ANNreg(x)

    # final loss (MSE) -
    testloss = (predictions-y).pow(2).mean()   #loss functions goes down over training epochs

    #plot the loss function
    # plt.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=.1)
    # plt.plot(numepochs, testloss.detach(), 'ro')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Final loss = %g' %testloss.item())
    # plt.show()

   #  print(testloss) # loss value and gradient info


    # now plot the predicted data observed data .95 very good
    plt.plot(x,y, 'bo', label='Real data')
    plt.plot(x, predictions.detach(), 'rs', label='Predictions')  #detaching the number fro all othr info
    plt.title(f'prediction-data r={np.corrcoef(y.T, predictions.detach().T)[0,1]:.2f}')
    plt.legend()
    plt.show()









