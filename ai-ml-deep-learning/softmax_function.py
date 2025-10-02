import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

'''
Prof. LaTonya Lewis
Softmax Function
The softmax function converts raw scores (logits) into a probability distribution across K classes, 
where each output is between 0 and 1, and all outputs sum to 1.
'''

# do manually in numpy
#list of numbers
z = [1,2,3]

#using softmax function to compute the softmax result
num = np.exp(z) #take natural exponent of each number - numerator
denominator = np.sum(np.exp(z)) #denominator
sigma = num/denominator  #output of the softmax function
print(sigma)
print(np.sum(sigma)) #sum over all the ouput is one (1)

#repeat with random integers
z = np.random.randint(-5, high=15, size=25) #twenty-five randon numbers from -5 through 15
print(z)

#repeating,  again with random numbers
#using softmax function to compute the softmax result
num = np.exp(z) #take natural exponent of each number - numerator
denominator = np.sum(np.exp(z)) #denominator
sigma = num/denominator  #output of the softmax function
print(sigma)
print(np.sum(sigma)) #sum over all the ouput is one (1)

#compare
plt.plot(z,sigma, 'ko')
plt.xlabel('Original number (z)')
plt.ylabel('Softmaxified $\sigma$')
plt.yscale('log') #logarithmic scale, vs original linear scale
plt.title('$\sum\sigma$ = %g' %np.sum(sigma))
plt.show()

#USE PYTORCH - more involved
#create an instance of the softmax activation class
softfunc = nn.Softmax(dim=0)

#apply the data to that function
sigmaT = softfunc(torch.Tensor(z)) #converting list into a pytorch tensor then put in softmax function

print(sigmaT)

type(torch.Tensor(z)) #transforming list into python torch.tensor

plt.plot(sigma,sigmaT, 'ko')
plt.xlabel('Manual softmax')
plt.ylabel('Pytorch nn.Software')
plt.title(f'The two methods correlate at r={np.corrcoef(sigma, sigmaT)[0,1]}')
plt.show()



