import numpy as np

'''
Professor Lewis
Compute the mean and variance using numpy
These are the quantities used for Normalization
'''

#create a list of numbers to compute the mean and variance

x = [1,2,4,6,5,4,0]
n = len(x)

'''
The mean is just the average — you add up all the numbers and divide by how many there are.

In simple terms: the mean tells you the “center” or the typical value in a set of numbers.
'''
# compute the mean
mean1 = np.mean(x)  #use numpy function
mean2 = np.sum(x) /n   #use formula explicitly

# print
print(mean1)
print(mean2)

'''
Variance tells us the average of the squared differences from the mean.
Standard deviation is just the square root of the variance.

In simple terms: they measure the same idea (spread of data), but variance is in squared units, 
while standard deviation brings it back to the original units so it’s easier to interpret.

The standard deviation tells us how spread out the numbers are from the average (mean).
A small standard deviation means most numbers are close to the mean.
A large standard deviation means the numbers are more spread out.
In simple terms: it measures how much the data “varies” from the center.
'''
# variance
var1 = np.var(x)  #use numpy function, uses a different degree of freedom of 0 , so we must set to ddof=1 for (1/(n-1))
var2 = (1/(n-1))* np.sum((x-mean1)**2)
print(var1)
print(var2)

var3 = np.var(x, ddof=1) # setting the degree of freedom to 1, ddof=1
#var2 and var3 are the same
print(var3)
print(var2)