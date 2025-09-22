import numpy as np
import matplotlib.pyplot as plt

'''
Prof. LaTonya Lewis
Sampling Variability 
refers to different samples from the same population can
have different values of the same measurement
the results (like the mean or proportion) will vary a little each time.
Law of Large Numbers - Averaging together many samples
will approximate the population mean, take many samples


In simple terms: it’s the “natural differences” you see when repeating a survey or experiment
with different groups of people or items, even though they all come from the same larger population.
'''

# create a list of numbers to compute the mean and variance of
x = [1,2,4,6,5,4,0,-4,5,-2,6,10,-9,1,3,-6]  #population of the height of people as example
n = len(x)

#compute the populaton mean
popmean = np.mean(x)

#compute the sample mean
sample = np.random.choice(x, size=5, replace=True)  #randomly pick five people from our random sample
sampmean = np.mean(sample)

#print
print(popmean)  #population mean
print(sampmean)  #sample mean of size five number

#Compute lots of sample means
# number of experiments to run, larger samples - better
nExperiments = 10000

#run the experiment
sampleMeans = np.zeros(nExperiments)
for i in range(nExperiments):
    # step 1: draw a sample
    sample = np.random.choice(x, size=5, replace=True)  #generate 10000 random samples

    # step 2 compute its mean
    sampleMeans[i] = np.mean(sample) #then compute and store the mean

# show results as a histogram
plt.hist(sampleMeans, bins=40, density=True)
plt.plot([popmean, popmean], [0,.3], 'm--')
plt.ylabel('Count')
plt.xlabel('Sample mean')
plt.show()


    #
