import numpy as np
import matplotlib.pyplot as plt

#dfine a set of points to evelauate
#plot 200 linear spaced number between .0001 and 1
x = np.linspace(.0001,1,200)

#computer their log
logx = np.log(x)
print(logx)

#plot them
fig = plt.figure(figsize=(10,4))

#increase font size .
plt.rcParams.update({'font.size':15})

plt.plot(x,logx, 'ks-', markerfacecolor='w')
plt.xlabel('x')
plt.ylabel('log(x)')
plt.show()


# demonstration that log and exp are inverses

#redefine with fewer points 20
x = np.linspace(.0001,1,20)

#log and exp
logx = np.log(x)  #log x
expx = np.exp(x)   # e to the x

#plot
plt.plot(x,x, color=[.8,.8,.8])
plt.plot(x,np.exp(logx), 'o', markersize=8)  #plottin natural exp of the log of x
plt.plot(x, np.log(expx), 'x', markersize=8)  #plotting the log of the natural exp of x
plt.xlabel('x')
plt.ylabel('f(g(x))')
plt.legend(['unity', 'exp(log(x))', 'log(exp(x))'])
plt.show()
