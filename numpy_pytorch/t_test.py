import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats  #use for T-test

'''
Prof. LaTonya Lewis
T-test - to test the prediction or accuracy of one model architecture to another
'''

# parameters
n1 = 30 #samples in dataset 1
n2 = 40 #samples in dataset 2
mu1 = 1 #population mean in dataset 1
mu2 = 1 #population mean in dataset 2

# generate the data
data1 = mu1 +  np.random.randn(n1)
data2 = mu2 + np.random.randn(n2)

#plot
#np.zeros(n1) → makes a list of zeros [0, 0, 0, …] of length n1.
#The zeros put all of data1’s points at x = 0 (first column on the plot).
#np.ones(n2) → makes a list of ones [1, 1, 1, …] of length n2.
#The ones put all of data2’s points at x = 1 (second column on the plot).
#So zeros and ones here are just telling Matplotlib where to line up your two groups of data on the x-axis.

plt.plot(np.zeros(n1), data1, 'ro',markerfacecolor='w', markersize=14)
plt.plot(np.ones(n2), data2, 'bs',markerfacecolor='w', markersize=14)
plt.xlim([-1,2])
plt.xticks([0,1], labels=['Group 1', 'Group 2'])
plt.show()

#so, the question is , is the difference between the mean of each data sample significantly statistical difference between both groups?
#we can test this using the T-test

#t-t4est via stats package
# _ind = independent samples
#the accuracies/predictions from data1 and data2 , data2 has a greater mean than data1 -
#t-test formula portion subtracting mean of both x - y, so you may get a negative depending on which data is first the t and p value are stored in tuple
t,p = stats.ttest_ind(data1, data2)
print(t)
print(p)

#common way to show visualize the data - t-t4est results in a plot
fig = plt.figure(figsize=(10,4))
plt.rcParams.update({'font.size':12}) #change the font size  , updatibg python parameters rcParams
plt.plot(0 + np.random.randn(n1)/15, data1, 'ro', markerfacecolor='w', markersize=14)  #x values 15
plt.plot(1 + np.random.randn(n2)/15, data2, 'bs', markerfacecolor='w', markersize=14)
plt.xlim([-1,2])
plt.xticks([0,1], labels=['Group 1', 'Group 2'])

#set the title to include the t-value and p-value
plt.title(f't= {t:2f}, p={p:3f}')
plt.show()








