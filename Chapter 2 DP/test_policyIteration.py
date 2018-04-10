import numpy as np
from prettytable import PrettyTable
from dynamicProgramming import *

# initialization (random policy)
initial_policy = np.zeros([16,4])
initial_policy[1:15,:] = 0.25

# transition probabilities p(s,a,s')
P = np.zeros([16,4,16])
P[0,:,0] = 1
P[15,:,15] = 1

# for action = 0 (left)
for i in [1,2,3,5,6,7,9,10,11,13,14]:
    P[i,0,i-1] = 1
for i in [4,8,12]:
    P[i,0,i] = 1

# for action = 1 (up)
for i in range(3):
    P[i+1,1,i+1] = 1
for i in range(11):
    P[i+4,1,i] = 1

# for action = 2 (right)
for i in [1,2,4,5,6,8,9,10,12,13,14]:
    P[i,2,i+1] = 1
for i in [3,7,11]:
    P[i,2,i] = 1

# for action = 3 (down)
for i in range(11):
    P[i+1,3,i+5] = 1
for i in range(3):
    P[i+12,3,i+12]=1
    
# immediate reward R(s,s',a)
R = np.zeros([16,4,16])
for i in range(14):
    R[i+1,:,:] = -1

# set other parameters
theta = 1e-6
gamma = 1

optimal_policy, optimal_value = policyIteration(P,R,gamma,theta,initial_policy)

print('optimal policy')
table = PrettyTable(['state', 'left','up','right','down'])
for i in range(14):
    table.add_row([i+1, "%.2f" % optimal_policy[i+1,0],
                        "%.2f" % optimal_policy[i+1,1],
                        "%.2f" % optimal_policy[i+1,2],
                        "%.2f" % optimal_policy[i+1,3]])
print(table)

#Policy Iteration:  1
#Policy Iteration:  2
#Policy Iteration:  3
#optimal policy
#+-------+------+------+-------+------+
#| state | left |  up  | right | down |
#+-------+------+------+-------+------+
#|   1   | 1.00 | 0.00 |  0.00 | 0.00 |
#|   2   | 1.00 | 0.00 |  0.00 | 0.00 |
#|   3   | 0.50 | 0.00 |  0.00 | 0.50 |
#|   4   | 0.00 | 1.00 |  0.00 | 0.00 |
#|   5   | 0.50 | 0.50 |  0.00 | 0.00 |
#|   6   | 0.25 | 0.25 |  0.25 | 0.25 |
#|   7   | 0.00 | 0.00 |  0.00 | 1.00 |
#|   8   | 0.00 | 1.00 |  0.00 | 0.00 |
#|   9   | 0.25 | 0.25 |  0.25 | 0.25 |
#|   10  | 0.00 | 0.00 |  0.50 | 0.50 |
#|   11  | 0.00 | 0.00 |  0.00 | 1.00 |
#|   12  | 0.00 | 0.50 |  0.50 | 0.00 |
#|   13  | 0.00 | 0.00 |  1.00 | 0.00 |
#|   14  | 0.00 | 0.00 |  1.00 | 0.00 |
#+-------+------+------+-------+------+