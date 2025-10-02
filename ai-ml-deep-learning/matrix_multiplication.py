import numpy as np
import torch

"""
Professor Lewis
Matrix multiplication 
"""

#USING NUMPY
#create som random matrices with varios sizes
A = np.random.randn(3,4)
B = np.random.randn(4,5)
C = np.random.randn(3,7)

#Matrix multiplication in Pythom the @ symbol does matrix multiplication
# np.multiply(A,B) is the same as A@B
# print(np.round(A@B,2)), print(' ')  #inner value same , runs successfully
#print(np.round(A@C, 2)), print('')  #error - inner dimensions different give error, need to transpose C
#print(np.round(B@C, 2)), print('') #error - inner dimensions different - unable to transpose
#print(np.round(C.T@A,2)) #transpose C 7x3  and A is still 3 x4 - now inner dimensions are same, the matrix product is 7x4 outer dimensions

#USIN PYTORCH

#create some random matrices
A = torch.randn(3,4)
B = torch.randn(4,5)
C1 = np.random.randn(4,7)
C2 = torch.tensor(C1, dtype=torch.float) #convert numpy matrix into a pytorch tensor, and the tensor will be of float type

#matrix multiplication
#print(np.round(A@B,2)), print('')
#print(np.round(A@B.T,2)), print('') # error shapes are different , inner deminsion, since B was transposed
#print(np.round(A@C1,2)), print('')#apply matrix multiplication on different types, numpy and pytorch
print(np.round(A@C2,2))
