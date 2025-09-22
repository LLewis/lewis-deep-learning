import numpy as np
import matplotlib.pyplot as plt

from IPython import display
#display.set_matplotlib_format('svg')  #draws figures in vector format instead of pixel format
display.display_svg()



#Grradient Descent 1D
#find the minimum of a function:  f(x) = 3x**2 - 3x + 4
#Emperically identify the local min of function

#function (as a function)
def fx(x):
    return 3*x**2 - 3*x + 4

#derivative function
def deriv(x):
    return 6*x -3


#defin a range for 'x' -2 to 2 in 2001 steps
x = np.linspace(-1,2,2001)

#plotting
plt.plot(x, fx(x), x, deriv(x)) #plotting x the func and x the deriv
plt.xlim(x[[0,-1]])
plt.gray()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['y', 'dy'])
plt.show()

#random starting point
localmin = np.random.choice(x,1) #from the vector x (on line 22 above), choose one random value, vector 'x' are the points defined above
print(localmin) #display the estimate or choice for localmin

#learning parameters
learning_rate = .01
training_epoch = 100

# run through training that implements the gradient descent algorithm
for i in range(training_epoch):
    grad = deriv(localmin) #compute the gradient as the derivative of the function at our localmin point (we use our random choice guess
    localmin = localmin - learning_rate*grad # adjust the localmin, the learning rate, the derivative is scaled by learning rate

print(localmin)  #Looking for 0.5 or close on 'x' axis

#plot the results
plt.plot(x, fx(x), x, deriv(x)) #plotting x the func and x the deriv
plt.plot(localmin, deriv(localmin), 'ro')
plt.plot(localmin, fx(localmin), 'ro')  #plots local minimum


plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)', 'df', 'f(x) min'])
plt.title('Emperical local minimum: %s' %localmin[0])
plt.show()

#complete remaining tomorrow Gradient 1D


