import numpy as np
import torch

#Using Numpy
#generate a few random numbers
np.random.randn(5)

#repeat after fixing the seed (an old method, but still widely used method)
#the seed “locks in” the randomness so your results are reproducible
np.random.seed(17) # random seed value that will reproduce the same state for 5 values each run, so the random() seed is being set to 17 an arbitrary number
print(np.random.randn(5))  # number of values in a list
print(np.random.randn(5))

#New seed mechanism in numpy
randseed1 = np.random.RandomState(17)
randseed2 = np.random.RandomState(20250917)

print()
print(randseed1.randn(5)) #same sequence
print(randseed2.randn(5)) #different from above, but same each time
print(randseed1.randn(5)) #same as two up
print(randseed2.randn(5)) #same as two up
print(np.random.randn(5))  # different every time  - not seeded by anything

# [ 0.27626589 -1.85462808  0.62390111  1.14531129  1.03719047]
# [ 1.88663893 -0.11169829 -0.36210134  0.14867505 -0.43778315]
#
# [ 0.27626589 -1.85462808  0.62390111  1.14531129  1.03719047]
# [ 0.04981175 -0.88972564  0.91969475 -1.39462755 -0.97862419]
# [ 1.88663893 -0.11169829 -0.36210134  0.14867505 -0.43778315]
# [ 0.19027906  1.8058817  -0.8215937  -0.67535105  0.14977134]
# [ 2.171257    1.15231025 -1.81881234 -0.13804934  0.53983961]


#Using Pytorch
torch.manual_seed(17)
print(torch.randn(5))

#Output
# tensor([-1.4135,  0.2336,  0.0340,  0.3499, -0.0145])


# torch's seed does not spread to numpy
print(np.random.randn(5))

#Output - shows pytorch random seed does not have in the same scope of numpy 
# [-1.77528229  1.31487654 -0.47344805 -1.0922299  -0.25002744]
