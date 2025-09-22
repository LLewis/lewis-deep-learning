import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
'''
Prof. LaTonya Lewis
Entropy - Claude Shannon - measures the surprise or uncertainty for a specific variables
When entropy is maximal probabilities are chance level, uncertain 
When entropy is minimal, when probabilities go to zero or one , there's more certainty or predictability ,(it is a cat or it's not a cate)
Cross-Entropy describes the relationship between two probability distributions

Entropy 
For a discrete probability distribution p(x):
Entropy measures the uncertainty (or average information content) in a probability distribution.

H(p)=−∑p(x)log p(x)
      x
      
Cross-Entropy
Given a true distribution p(x) and an estimated (model) distribution q(x):
Cross-entropy measures how well the estimated distribution q(x) predicts the true distribution p(x).

H(p,q)=−x∑p(x)log q(x)
         x
In machine learning, entropy often represents the baseline uncertainty in the data, 
while cross-entropy is used as a loss function to evaluate how close a model’s predictions are to the actual labels.
'''

#USING NUMPY
#Entropy
#probability of an event happening
x = [.25, .75] #.25 event happening, .75 event not happening

H = 0
for p in x:
    H -= p*np.log(p)
    print('Correct entropy: ' + str(H))

#another way to write explicitly out for N=2 events, instead of using the for loop
    #probability it is a cat p or it is not a cat (1-p)
H = - (p*np.log(p) + (1-p)* np.log(1-p))  # Binary Cross-Entropy loss function in deep learning
print('Correct Entropy: ' +str(H))


#Cross-Entropy
#All probabilities must sum to 1, essentially the same, using two events instead of one event
p = [1,0] #sum = 1
q = [.25, .75] #sum = 1
H= 0
for i in range(len(p)):
    H -= p[i] *np.log(q[i])
    print('Cross Entropy: ' + str(H))

#another way to write explicitly out for N=2 events, instead of using the for loop
H = - (p[0]*np.log(q[0]) + (p[1])* np.log(q[1]))  # Binary Cross-Entropy loss function in deep learning
print('Cross Entropy: ' + str(H))

#can simplify
H = -np.log(q[0])  #the zero cancels out
print('Manually simplified: ' + str(H))

#USING PYTORCH
 #inputs must be Tensors
#q_tensor model predicted probability is the picture a cat or isn't a cat
q_tensor = torch.Tensor(q)
# p_tensor category label, is a cat, isn't a cat
p_tensor = torch.Tensor(p)


print(F.binary_cross_entropy(q_tensor,p_tensor) ) #cross because two event and binary only two options, it is or isn't, must convert from list to tensor














