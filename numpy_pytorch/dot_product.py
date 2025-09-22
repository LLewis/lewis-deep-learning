import numpy as np
import torch

#Python --version 3.12
#python -m pip install --upgrade "numpy<2"`
#installed pytorch - brew install pytorch

#USING numpy to find the dot product between two vectors
#create two vectors datatype is an integer
nv1 = np.array([1,2,3,4])
nv2 = np.array([0,1,0,-1])


#use dot product function - do element wide multiplication 1x0+2x1+3x0+4x-1 = -2
#vectors must be same dimension to calculate dot product -
#test by removing one element from array - get ValueError shapes not aligned message
print(np.dot(nv1,nv2))

#use dot product via computation
print(np.sum(nv1*nv2))

#USING pytorch
#create two vectors  - datatype is a tensor
tv1 = torch.tensor([1,2,3,4])
tv2 = torch.tensor([0,1,0,-1])

# dot product via function
print(torch.dot(tv1,tv2))

#dot product via computation
print(torch.sum(tv1*tv2))